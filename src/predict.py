import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

from src.data import find_occurrences, piece_index, split_pieces, strip_punct, token_index, word_pattern
from src.models import (
    Item, candidate_features, collate, encoder_features, get_normalizer, load_checkpoint, seq2seq_inputs,
)

SENTENCE_END = re.compile(r"(?<=[।?!\n])")
CONTEXT_WORDS = 32  # words kept on each side of the target when a single "sentence" is very long


@dataclass
class Prediction:
    pronunciation: str
    confidence: float
    aligned: bool = True
    output: str = ""
    generated_tokens: int = 0


class GreedyLogProbs(LogitsProcessor):
    """Records the log-probability of each greedy choice (max logit - logsumexp) while generating. Keeping all
    scores (output_scores) instead needs steps x batch x vocab floats: ~2.4 GB for mT5's 250k vocab at batch 64."""

    def __init__(self):
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores32 = scores.float()
        self.steps.append(scores32.max(-1).values - scores32.logsumexp(-1))
        return scores


def roundtrip(tokenizer: PreTrainedTokenizerBase, text: str) -> str:
    return tokenizer.decode(tokenizer(text)["input_ids"], skip_special_tokens=True)


def sentence_item(text: str, start: int, end: int) -> Item:
    """The item for the occurrence text[start:end], cut to the sentence that contains it."""
    offset = 0
    for piece in SENTENCE_END.split(text):
        if offset <= start < offset + len(piece):
            return crop(piece, start - offset, end - offset)
        offset += len(piece)
    raise ValueError(f"position {start} is outside the text")


def crop(sentence: str, start: int, end: int) -> Item:
    left, right = sentence[:start].split(" "), sentence[end:].split(" ")
    if len(left) <= CONTEXT_WORDS and len(right) <= CONTEXT_WORDS:
        return sentence, start, end, sentence[start:end]
    left = " ".join(left[-CONTEXT_WORDS:])
    return left + sentence[start:end] + " ".join(right[:CONTEXT_WORDS]), len(left), len(left) + end - start, sentence[start:end]


class Predictor:
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, hconfig: dict, device: str | torch.device, half: bool):
        self.model = (model.half() if half else model).to(device).eval()
        self.tokenizer = tokenizer
        self.hconfig = hconfig
        self.device = torch.device(device)
        self.candidates: dict[str, list[str]] = hconfig["candidates"]
        self.marker = tuple(hconfig["marker"]) if hconfig["marker"] else None
        self.norm = get_normalizer(hconfig["normalize"], hconfig.get("strip_zwnj", False))
        self.pattern = word_pattern(list(self.candidates))
        self.kind = hconfig["model_type"]
        if self.kind == "seq2seq":
            # Predictions and references are compared after the same normalizer + tokenizer round trip.
            # Tokenizers that cannot represent the ZWNJ get it removed from inputs and targets (strip_zwnj);
            # a predicted target is mapped back to the original candidate, which keeps its ZWNJ.
            self.canon = {w: [self.canonical(c) for c in cands] for w, cands in self.candidates.items()}

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None, precision: str | None = None) -> "Predictor":
        model, tokenizer, hconfig = load_checkpoint(checkpoint)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        precision = precision or (hconfig["precision"] if device != "cpu" else "fp32")
        return cls(model, tokenizer, hconfig, device, precision == "fp16")

    def canonical(self, text: str) -> str:
        return strip_punct(roundtrip(self.tokenizer, self.norm(text)))

    def encode(self, items: list[Item]) -> dict[str, torch.Tensor]:
        hc = self.hconfig
        if self.kind == "seq2seq":
            batch = self.tokenizer(seq2seq_inputs(items, self.marker, self.norm), padding=True, truncation=True,
                                   max_length=hc["max_length"], return_tensors="pt")
        else:
            build = candidate_features if self.kind == "candidate" else encoder_features
            batch = collate(build(self.tokenizer, items, self.candidates, self.marker, self.norm, hc["max_length"]),
                            self.tokenizer.pad_token_id)
        return {k: v.to(self.device) for k, v in batch.items()}

    def forward(self, batch: dict[str, torch.Tensor]):
        with torch.inference_mode():
            if self.kind == "seq2seq":
                recorder = GreedyLogProbs()
                sequences = self.model.generate(**batch, max_new_tokens=self.hconfig["max_length"], num_beams=1, do_sample=False,
                                                logits_processor=LogitsProcessorList([recorder]))
                return sequences, torch.stack(recorder.steps, 1)
            return self.model(**batch)["logits"]

    def decode(self, items: list[Item], outputs) -> list[Prediction]:
        if self.kind != "seq2seq":
            confidence, index = outputs.float().softmax(-1).max(-1)
            return [Prediction(self.candidates[w][i], c) for (*_, w), i, c in zip(items, index.tolist(), confidence.tolist())]
        sequences, logprobs = outputs
        mask = (sequences[:, -logprobs.size(1):] != self.tokenizer.pad_token_id).float()
        mean_logprob = (logprobs * mask).sum(1) / mask.sum(1).clamp(min=1)
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        predictions = []
        for text, item, lp, n in zip(texts, items, mean_logprob.tolist(), mask.sum(1).tolist()):
            prediction = self.from_output(text, item)
            prediction.confidence, prediction.generated_tokens = math.exp(lp), int(n)
            predictions.append(prediction)
        return predictions

    def from_output(self, text: str, item: Item) -> Prediction:
        sentence, start, _, word = item
        if self.hconfig["output"] == "target":
            raw, aligned = strip_punct(text), True
        else:
            tokens, idx, piece = text.split(), token_index(sentence, start), piece_index(sentence, start)
            aligned = len(tokens) == len(sentence.split())
            pieces = split_pieces(tokens[idx]) if idx < len(tokens) else []
            raw = strip_punct(pieces[piece]) if piece < len(pieces) else ""
        canon = self.canon[word]
        pronunciation = self.candidates[word][canon.index(raw)] if raw in canon else raw
        return Prediction(pronunciation, 0.0, aligned, text)

    def predict_items(self, items: list[Item], batch_size: int = 64) -> list[Prediction]:
        predictions = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            predictions += self.decode(chunk, self.forward(self.encode(chunk)))
        return predictions

    # Single calls go through the batch methods; a batch method sends all occurrences of all its inputs through
    # predict_items, one model call per batch_size occurrences.

    def check_word(self, word: str) -> None:
        if word not in self.candidates:
            raise ValueError(f"{word!r} is not one of the {len(self.candidates)} known heteronyms")

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Reading of the first whole-word occurrence of each word in its sentence."""
        items = []
        for sentence, word in pairs:
            self.check_word(word)
            occurrences = find_occurrences(sentence, word_pattern([word]))
            if not occurrences:
                raise ValueError(f"{word!r} does not occur as a whole word in {sentence!r}")
            start, end, _ = occurrences[0]
            items.append(sentence_item(sentence, start, end))
        return [p.pronunciation for p in self.predict_items(items)]

    def predict(self, sentence: str, word: str) -> str:
        return self.predict_batch([(sentence, word)])[0]

    def predict_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        """Reading of the one heteronym written between the marks in each text, e.g. "আগে কর কাজটা, তারপর আবার [কর]।".
        Only the sentence that contains the mark is given to the model, as in phoneticize."""
        items = []
        for text in texts:
            start = text.find(marks[0])
            end = text.find(marks[1], start + len(marks[0])) if start >= 0 else -1
            if start < 0 or end < 0:
                raise ValueError(f"mark the heteronym as {marks[0]}word{marks[1]}: {text!r}")
            word = text[start + len(marks[0]):end]
            self.check_word(word)
            plain = text[:start] + word + text[end + len(marks[1]):]
            items.append(sentence_item(plain, start, start + len(word)))
        return [p.pronunciation for p in self.predict_items(items)]

    def predict_marked(self, text: str, marks: tuple[str, str] = ("[", "]")) -> str:
        return self.predict_marked_batch([text], marks)[0]

    def phoneticize_detailed_batch(self, texts: list[str]) -> list[tuple[str, list[dict]]]:
        items, owners = [], []  # every occurrence in every text, predicted together
        for t, text in enumerate(texts):
            offset = 0
            for piece in SENTENCE_END.split(text):
                for start, end, _ in find_occurrences(piece, self.pattern):
                    items.append(crop(piece, start, end))
                    owners.append((t, offset + start, offset + end))
                offset += len(piece)
        spans: list[list] = [[] for _ in texts]
        for item, (t, start, end), p in zip(items, owners, self.predict_items(items)):
            detail = {"word": item[3], "position": start, "prediction": p.pronunciation or item[3], "confidence": round(p.confidence, 4)}
            spans[t].append((start, end, detail))
        out = []
        for text, found in zip(texts, spans):
            for start, end, d in reversed(found):
                text = text[:start] + d["prediction"] + text[end:]
            out.append((text, [d for *_, d in found]))
        return out

    def phoneticize_detailed(self, text: str) -> tuple[str, list[dict]]:
        return self.phoneticize_detailed_batch([text])[0]

    def phoneticize_batch(self, texts: list[str]) -> list[str]:
        return [text for text, _ in self.phoneticize_detailed_batch(texts)]

    def phoneticize(self, text: str) -> str:
        return self.phoneticize_batch([text])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    text = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    predictor = Predictor.load(args.checkpoint, args.device)
    predictor.phoneticize(text)  # warm-up
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    phonetic, details = predictor.phoneticize_detailed(text)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(phonetic)
    print("\nword\tposition\tprediction\tconfidence")
    for d in details:
        print(f"{d['word']}\t{d['position']}\t{d['prediction']}\t{d['confidence']:.4f}")
    print(f"\n{len(details)} heteronyms, {elapsed:.1f} ms on {predictor.device.type} ({args.checkpoint})")


if __name__ == "__main__":
    main()
