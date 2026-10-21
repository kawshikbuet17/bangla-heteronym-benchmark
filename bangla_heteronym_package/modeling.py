import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from .text import mark

N_SLOTS = 3  # at most 3 readings per word
MASKED = -1e4


class EncoderClassifier(nn.Module):
    """Encoder + linear layer on the opening marker token (or the target's mean / [CLS]); one score per reading."""

    def __init__(self, encoder, head: str, n_outputs: int = N_SLOTS):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.out = nn.Linear(encoder.config.hidden_size, n_outputs)

    def score(self, input_ids, attention_mask, anchor, target_mask, token_type_ids):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        if self.head == "marker":
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), anchor]
        elif self.head == "cls":
            pooled = hidden[:, 0]
        else:  # mean of the target's tokens
            mask = target_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(pooled).float()

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None):
        return self.score(input_ids, attention_mask, anchor, target_mask, token_type_ids).masked_fill(~cand_mask, MASKED)


class CandidateScorer(EncoderClassifier):
    """Scores (sentence, candidate reading) pairs, one score per candidate."""

    def __init__(self, encoder, head: str):
        super().__init__(encoder, head, n_outputs=1)

    def forward(self, input_ids, attention_mask, anchor, target_mask, cand_mask, token_type_ids=None):
        b, c = input_ids.shape[:2]
        flat = [None if t is None else t.reshape(b * c, *t.shape[2:]) for t in (input_ids, attention_mask, anchor, target_mask, token_type_ids)]
        scores = F.pad(self.score(*flat).view(b, c), (0, N_SLOTS - c), value=MASKED)
        return scores.masked_fill(~cand_mask, MASKED)


def load_checkpoint(path):
    """Loads a benchmark checkpoint folder: weights, tokenizer and heteronym_config.json (with the word list)."""
    path = Path(path)
    config = json.loads((path / "heteronym_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    if config["model_type"] == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(path), tokenizer, config
    encoder = AutoModel.from_pretrained(path)
    if len(tokenizer) > encoder.get_input_embeddings().num_embeddings:
        encoder.resize_token_embeddings(len(tokenizer))
    model = CandidateScorer(encoder, config["head"]) if config["model_type"] == "candidate" else EncoderClassifier(encoder, config["head"])
    model.out.load_state_dict(load_file(path / "head.safetensors"))
    return model, tokenizer, config


def _features(enc, i: int, seq: int, anchor_char: int, span: tuple[int, int]) -> dict:
    targets = {enc.char_to_token(i, c, sequence_index=seq) for c in range(*span)} - {None}
    feature = {k: enc[k][i] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
    feature["anchor"] = enc.char_to_token(i, anchor_char, sequence_index=seq) or 0
    feature["target_mask"] = [int(t in targets) for t in range(len(feature["input_ids"]))]
    return feature


def encoder_batch(tokenizer, items, config, norm, pad_id: int) -> dict:
    """One padded batch for an encoder (input: word [SEP] marked sentence) or a candidate scorer (input: marked
    sentence [SEP] candidate, one row per candidate)."""
    candidates = config["candidates"]
    marked = [mark(s, a, b, config["marker"], norm) for s, a, b, _ in items]
    if config["model_type"] == "candidate":
        pairs = [(i, text, norm(c)) for i, ((text, *_), (*_, w)) in enumerate(zip(marked, items)) for c in candidates[w]]
        enc = tokenizer([t for _, t, _ in pairs], [c for *_, c in pairs], truncation="only_first", max_length=config["max_length"])
        rows = [[] for _ in items]
        for j, (i, *_) in enumerate(pairs):
            rows[i].append(_features(enc, j, 0, marked[i][1], marked[i][2:]))
    else:
        enc = tokenizer([norm(w) for *_, w in items], [t for t, *_ in marked], truncation="only_second", max_length=config["max_length"])
        rows = [[_features(enc, i, 1, m[1], m[2:])] for i, m in enumerate(marked)]
    n = max(len(r) for r in rows)
    rows = [r + [r[0]] * (n - len(r)) for r in rows]  # pad the candidate count by repeating the first pair (masked)
    length = max(len(p["input_ids"]) for r in rows for p in r)
    pad = lambda values, fill: values + [fill] * (length - len(values))
    batch = {key: torch.tensor([[pad(p[key], fill) for p in r] for r in rows])
             for key, fill in (("input_ids", pad_id), ("attention_mask", 0), ("token_type_ids", 0), ("target_mask", 0)) if key in rows[0][0]}
    batch["anchor"] = torch.tensor([[p["anchor"] for p in r] for r in rows])
    if config["model_type"] != "candidate":
        batch = {k: v.squeeze(1) for k, v in batch.items()}
    batch["cand_mask"] = torch.tensor([[k < len(candidates[w]) for k in range(N_SLOTS)] for *_, w in items])
    return batch
