"""Model classes, input features and checkpoint loading (inference only)."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

from .text import SEQ2SEQ_PREFIX, Item, Marker, Normalizer, build_input

N_SLOTS = 3
MASKED = -1e4
HCONFIG_FILE = "heteronym_config.json"
HEAD_FILE = "head.safetensors"


def pool(hidden: torch.Tensor, anchor: torch.Tensor, target_mask: torch.Tensor, head: str) -> torch.Tensor:
    if head == "marker":
        return hidden[torch.arange(hidden.size(0), device=hidden.device), anchor]
    if head == "cls":
        return hidden[:, 0]
    mask = target_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class EncoderClassifier(nn.Module):
    """Encoder + dropout + linear on one pooled vector; logits of non-existing readings are masked."""

    def __init__(self, encoder: nn.Module, head: str, dropout: float, n_outputs: int = N_SLOTS):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(encoder.config.hidden_size, n_outputs)

    def score(self, input_ids, attention_mask, anchor, target_mask, token_type_ids) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return self.out(self.dropout(pool(hidden, anchor, target_mask, self.head))).float()

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None):
        return self.score(input_ids, attention_mask, anchor, target_mask, token_type_ids).masked_fill(~cand_mask, MASKED)


class CandidateScorer(EncoderClassifier):
    """One (sentence, candidate) pair per candidate, one score each, softmax over the word's candidates."""

    def __init__(self, encoder: nn.Module, head: str, dropout: float):
        super().__init__(encoder, head, dropout, n_outputs=1)

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None):
        b, c = input_ids.shape[:2]
        flat = [None if t is None else t.reshape(b * c, *t.shape[2:]) for t in (input_ids, attention_mask, anchor, target_mask, token_type_ids)]
        scores = self.score(*flat).view(b, c)
        return F.pad(scores, (0, N_SLOTS - c), value=MASKED).masked_fill(~cand_mask, MASKED)


def _pair_features(enc, i: int, seq: int, anchor_char: int, span: tuple[int, int]) -> dict:
    anchor = enc.char_to_token(i, anchor_char, sequence_index=seq)
    targets = {enc.char_to_token(i, c, sequence_index=seq) for c in range(*span)} - {None}
    feature = {k: enc[k][i] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
    feature["anchor"] = 0 if anchor is None else anchor
    feature["target_mask"] = [int(t in targets) for t in range(len(feature["input_ids"]))]
    return feature


def encoder_features(tokenizer: PreTrainedTokenizerBase, items: list[Item], candidates: dict[str, list[str]],
                     marker: Marker, norm: Normalizer, max_length: int) -> list[dict]:
    built = [build_input(s, a, b, marker, norm) for s, a, b, _ in items]
    words = [norm(w) for *_, w in items]
    enc = tokenizer(words, [t for t, *_ in built], truncation="only_second", max_length=max_length)
    features = []
    for i, ((_, anchor, ts, te), (*_, word)) in enumerate(zip(built, items)):
        feature = _pair_features(enc, i, 1, anchor, (ts, te))
        feature["cand_mask"] = [k < len(candidates[word]) for k in range(N_SLOTS)]
        features.append(feature)
    return features


def candidate_features(tokenizer: PreTrainedTokenizerBase, items: list[Item], candidates: dict[str, list[str]],
                       marker: Marker, norm: Normalizer, max_length: int) -> list[dict]:
    built = [build_input(s, a, b, marker, norm) for s, a, b, _ in items]
    firsts, seconds, owners = [], [], []
    for i, ((text, *_), (*_, word)) in enumerate(zip(built, items)):
        for cand in candidates[word]:
            firsts.append(text)
            seconds.append(norm(cand))
            owners.append(i)
    enc = tokenizer(firsts, seconds, truncation="only_first", max_length=max_length)
    features = [{"cand_mask": [k < len(candidates[w]) for k in range(N_SLOTS)], "pairs": []} for *_, w in items]
    for j, i in enumerate(owners):
        _, anchor, ts, te = built[i]
        features[i]["pairs"].append(_pair_features(enc, j, 0, anchor, (ts, te)))
    return features


def seq2seq_inputs(items: list[Item], marker: Marker, norm: Normalizer) -> list[str]:
    return [SEQ2SEQ_PREFIX + build_input(s, a, b, marker, norm)[0] for s, a, b, _ in items]


def _pad(rows: list[list[int]], length: int, value: int) -> list[list[int]]:
    return [row + [value] * (length - len(row)) for row in rows]


def collate(features: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pads encoder or candidate features into one batch. Candidate pairs are padded to the batch's largest
    candidate count by repeating the first pair (its score is masked out)."""
    candidate = "pairs" in features[0]
    rows = [f["pairs"] for f in features] if candidate else [[f] for f in features]
    n = max(len(row) for row in rows)
    rows = [row + [row[0]] * (n - len(row)) for row in rows]
    length = max(len(p["input_ids"]) for row in rows for p in row)
    batch = {}
    for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("token_type_ids", 0), ("target_mask", 0)):
        if key in rows[0][0]:
            batch[key] = torch.tensor([_pad([p[key] for p in row], length, value) for row in rows])
    batch["anchor"] = torch.tensor([[p["anchor"] for p in row] for row in rows])
    if not candidate:
        batch = {k: v.squeeze(1) for k, v in batch.items()}
    batch["cand_mask"] = torch.tensor([f["cand_mask"] for f in features], dtype=torch.bool)
    return batch


def load_checkpoint(path: str | Path) -> tuple[nn.Module, PreTrainedTokenizerBase, dict]:
    """Loads a checkpoint folder written by the benchmark (weights, tokenizer and heteronym_config.json)."""
    path = Path(path)
    if not (path / HCONFIG_FILE).exists():
        raise FileNotFoundError(f"{path} is not a heteronym checkpoint folder (no {HCONFIG_FILE})")
    hconfig = json.loads((path / HCONFIG_FILE).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    if hconfig["model_type"] == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(path).eval(), tokenizer, hconfig
    encoder = AutoModel.from_pretrained(path)
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    cls = CandidateScorer if hconfig["model_type"] == "candidate" else EncoderClassifier
    model = cls(encoder, hconfig["head"], hconfig["dropout"])
    model.out.load_state_dict(load_file(path / HEAD_FILE))
    return model.eval(), tokenizer, hconfig
