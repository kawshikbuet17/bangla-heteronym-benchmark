"""HeteronymReader implemented with a trained checkpoint (encoder classifier, candidate scorer or seq2seq)."""
import math
import threading
from pathlib import Path

import torch
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

from .interface import HeteronymReader, Reading
from .modeling import candidate_features, collate, encoder_features, load_checkpoint, seq2seq_inputs
from .text import (
    SENTENCE_END, Item, crop, find_occurrences, get_normalizer, piece_index, sentence_item, split_pieces, strip_punct,
    token_index, word_pattern,
)


class _GreedyLogProbs(LogitsProcessor):
    """Records the log-probability of each greedy choice while generating (for the seq2seq confidence)."""

    def __init__(self):
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores32 = scores.float()
        self.steps.append(scores32.max(-1).values - scores32.logsumexp(-1))
        return scores


class HeteronymModelReader(HeteronymReader):
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, hconfig: dict, device: str, half: bool,
                 batch_size: int = 64):
        self.model = (model.half() if half else model).to(device).eval()
        self.tokenizer = tokenizer
        self.hconfig = hconfig
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._lock = threading.Lock()  # one model call at a time when threads share the reader
        self.candidates: dict[str, list[str]] = hconfig["candidates"]
        self.marker = tuple(hconfig["marker"]) if hconfig["marker"] else None
        self.norm = get_normalizer(hconfig["normalize"], hconfig.get("strip_zwnj", False))
        self.pattern = word_pattern(list(self.candidates))
        self.kind = hconfig["model_type"]
        if self.kind == "seq2seq":
            # A generated word is compared with each candidate after the same normalizer + tokenizer round trip,
            # then mapped back to the candidate, so outputs always keep their ZWNJ.
            self.canon = {w: [self._canonical(c) for c in cands] for w, cands in self.candidates.items()}

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None, precision: str | None = None,
             batch_size: int = 64) -> "HeteronymModelReader":
        """checkpoint: a checkpoint folder (absolute or relative path). device: "cuda", "cuda:1", "cpu" (default: cuda
        if available). precision: "fp16" or "fp32" (default: the training precision on a GPU, fp32 on a CPU)."""
        model, tokenizer, hconfig = load_checkpoint(checkpoint)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        precision = precision or (hconfig["precision"] if not device.startswith("cpu") else "fp32")
        return cls(model, tokenizer, hconfig, device, precision == "fp16", batch_size)

    @property
    def heteronyms(self) -> dict[str, list[str]]:
        return self.candidates

    # ---- public API (see interface.py)

    def readings_batch(self, texts: list[str]) -> list[list[Reading]]:
        items, owners = [], []  # every occurrence in every text, predicted together
        for t, text in enumerate(texts):
            offset = 0
            for piece in SENTENCE_END.split(text):
                for start, end, word in find_occurrences(piece, self.pattern):
                    items.append(crop(piece, start, end))
                    owners.append((t, offset + start, offset + end, word))
                offset += len(piece)
        results: list[list[Reading]] = [[] for _ in texts]
        for (t, start, end, word), (reading, confidence) in zip(owners, self._predict_items(items)):
            results[t].append(Reading(word, start, end, reading or word, round(confidence, 4)))
        return results

    def phoneticize_batch(self, texts: list[str]) -> list[str]:
        out = []
        for text, readings in zip(texts, self.readings_batch(texts)):
            for r in reversed(readings):
                text = text[:r.start] + r.reading + text[r.end:]
            out.append(text)
        return out

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        items = []
        for sentence, word in pairs:
            self._check_word(word)
            occurrences = find_occurrences(sentence, word_pattern([word]))
            if not occurrences:
                raise ValueError(f"{word!r} does not occur as a whole word in {sentence!r}")
            start, end, _ = occurrences[0]
            items.append(sentence_item(sentence, start, end))
        return [reading for reading, _ in self._predict_items(items)]

    def predict_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        items = []
        for text in texts:
            start = text.find(marks[0])
            end = text.find(marks[1], start + len(marks[0])) if start >= 0 else -1
            if start < 0 or end < 0:
                raise ValueError(f"mark the heteronym as {marks[0]}word{marks[1]}: {text!r}")
            word = text[start + len(marks[0]):end]
            self._check_word(word)
            plain = text[:start] + word + text[end + len(marks[1]):]
            items.append(sentence_item(plain, start, start + len(word)))
        return [reading for reading, _ in self._predict_items(items)]

    # ---- implementation

    def _check_word(self, word: str) -> None:
        if word not in self.candidates:
            raise ValueError(f"{word!r} is not one of the {len(self.candidates)} known heteronyms")

    def _canonical(self, text: str) -> str:
        return strip_punct(self.tokenizer.decode(self.tokenizer(self.norm(text))["input_ids"], skip_special_tokens=True))

    def _predict_items(self, items: list[Item]) -> list[tuple[str, float]]:
        """(reading, confidence) per item; one model call per batch_size items."""
        out = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            with self._lock:
                out += self._decode(chunk, self._forward(self._encode(chunk)))
        return out

    def _encode(self, items: list[Item]) -> dict[str, torch.Tensor]:
        hc = self.hconfig
        if self.kind == "seq2seq":
            batch = self.tokenizer(seq2seq_inputs(items, self.marker, self.norm), padding=True, truncation=True,
                                   max_length=hc["max_length"], return_tensors="pt")
        else:
            build = candidate_features if self.kind == "candidate" else encoder_features
            batch = collate(build(self.tokenizer, items, self.candidates, self.marker, self.norm, hc["max_length"]),
                            self.tokenizer.pad_token_id)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _forward(self, batch: dict[str, torch.Tensor]):
        with torch.inference_mode():
            if self.kind == "seq2seq":
                recorder = _GreedyLogProbs()
                sequences = self.model.generate(**batch, max_new_tokens=self.hconfig["max_length"], num_beams=1, do_sample=False,
                                                logits_processor=LogitsProcessorList([recorder]))
                return sequences, torch.stack(recorder.steps, 1)
            return self.model(**batch)

    def _decode(self, items: list[Item], outputs) -> list[tuple[str, float]]:
        if self.kind != "seq2seq":
            confidence, index = outputs.float().softmax(-1).max(-1)
            return [(self.candidates[w][i], c) for (*_, w), i, c in zip(items, index.tolist(), confidence.tolist())]
        sequences, logprobs = outputs
        mask = (sequences[:, -logprobs.size(1):] != self.tokenizer.pad_token_id).float()
        mean_logprob = (logprobs * mask).sum(1) / mask.sum(1).clamp(min=1)
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return [(self._from_output(text, item), math.exp(lp)) for text, item, lp in zip(texts, items, mean_logprob.tolist())]

    def _from_output(self, text: str, item: Item) -> str:
        """The generated word at the target's position, mapped back to its candidate when it matches one."""
        sentence, start, _, word = item
        if self.hconfig["output"] == "target":
            raw = strip_punct(text)
        else:
            tokens, idx, piece = text.split(), token_index(sentence, start), piece_index(sentence, start)
            pieces = split_pieces(tokens[idx]) if idx < len(tokens) else []
            raw = strip_punct(pieces[piece]) if piece < len(pieces) else ""
        canon = self.canon[word]
        return self.candidates[word][canon.index(raw)] if raw in canon else raw
