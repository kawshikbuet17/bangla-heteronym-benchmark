import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

from src.data import ZWNJ, Marker, Normalizer, build_input

N_SLOTS = 3
MASKED = -1e4
SEQ2SEQ_PREFIX = "phonetic: "
HCONFIG_FILE = "heteronym_config.json"
HEAD_FILE = "head.safetensors"

Item = tuple[str, int, int, str]  # sentence, target start, target end, word


def get_normalizer(enabled: bool, strip_zwnj: bool = False) -> Normalizer:
    base = lambda text: text
    if enabled:
        from normalizer import normalize as base  # only BanglaBERT/BanglaT5 need this package
    if not strip_zwnj:
        return base
    return lambda text: base(text).replace(ZWNJ, "")


def keeps_zwnj(tokenizer: PreTrainedTokenizerBase) -> bool:
    sample = "আশ্\u200cলো"
    return tokenizer.decode(tokenizer(sample)["input_ids"], skip_special_tokens=True) == sample


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

    def output(self, logits: torch.Tensor, cand_mask: torch.Tensor, labels: torch.Tensor | None) -> dict:
        logits = logits.masked_fill(~cand_mask, MASKED)
        if labels is None:
            return {"logits": logits}
        return {"loss": F.cross_entropy(logits, labels), "logits": logits}

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None, labels=None):
        return self.output(self.score(input_ids, attention_mask, anchor, target_mask, token_type_ids), cand_mask, labels)


class CandidateScorer(EncoderClassifier):
    """One (sentence, candidate) pair per candidate, one score each, softmax over the word's candidates."""

    def __init__(self, encoder: nn.Module, head: str, dropout: float):
        super().__init__(encoder, head, dropout, n_outputs=1)

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None, labels=None):
        b, c = input_ids.shape[:2]
        flat = [None if t is None else t.reshape(b * c, *t.shape[2:]) for t in (input_ids, attention_mask, anchor, target_mask, token_type_ids)]
        scores = self.score(*flat).view(b, c)
        return self.output(F.pad(scores, (0, N_SLOTS - c), value=MASKED), cand_mask, labels)


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
    """Pads encoder or candidate features. Candidate pairs are padded to the batch's largest candidate
    count by repeating the first pair (its score is masked out)."""
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
    if "labels" in features[0]:
        batch["labels"] = torch.tensor([f["labels"] for f in features])
    return batch


def build_model(hconfig: dict, tokenizer: PreTrainedTokenizerBase, source: str) -> nn.Module:
    if hconfig["model_type"] == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(source)
    encoder = AutoModel.from_pretrained(source)
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    cls = CandidateScorer if hconfig["model_type"] == "candidate" else EncoderClassifier
    return cls(encoder, hconfig["head"], hconfig["dropout"])


def save_checkpoint(model: nn.Module, tokenizer: PreTrainedTokenizerBase, hconfig: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hconfig["model_type"] == "seq2seq":
        model.save_pretrained(out_dir)
    else:
        model.encoder.save_pretrained(out_dir)
        save_file({k: v.contiguous() for k, v in model.out.state_dict().items()}, out_dir / HEAD_FILE)
    tokenizer.save_pretrained(out_dir)
    (out_dir / HCONFIG_FILE).write_text(json.dumps(hconfig, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> tuple[nn.Module, PreTrainedTokenizerBase, dict]:
    path = Path(path)
    hconfig = json.loads((path / HCONFIG_FILE).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = build_model(hconfig, tokenizer, str(path))
    if hconfig["model_type"] != "seq2seq":
        model.out.load_state_dict(load_file(path / HEAD_FILE))
    return model.eval(), tokenizer, hconfig
