import re
import warnings

import torch

from .interface import HeteronymReader
from .modeling import encoder_batch, load_checkpoint
from .text import BOUNDARY, get_normalizer, mark, sentence_item, strip_punct, target_piece, word_pattern

BATCH_SIZE = 64  # occurrences per model call
# Model types that run on padded batches: all 12 benchmark models and the candidate / target-only checkpoints.
BATCH_MODEL_TYPES = ("encoder", "candidate", "seq2seq")


class HeteronymModelReader(HeteronymReader):
    """HeteronymReader backed by a trained checkpoint (encoder classifier, candidate scorer or seq2seq model)."""

    def __init__(self, checkpoint_path, device: str | None = None):
        model, self.tokenizer, self.config = load_checkpoint(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        half = self.config["precision"] == "fp16" and not self.device.startswith("cpu")
        self.model = (model.half() if half else model).to(self.device).eval()
        self.heteronyms: dict[str, list[str]] = self.config["candidates"]  # word -> its readings
        self.pattern = word_pattern(self.heteronyms)
        self.norm = get_normalizer(self.config["normalize"], self.config.get("strip_zwnj", False))
        self.supports_batch = self.config["model_type"] in BATCH_MODEL_TYPES and self.config.get("batch", True)
        self.batch_size = BATCH_SIZE if self.supports_batch else 1
        if not self.supports_batch:
            warnings.warn(f"{checkpoint_path} does not support batch inference; the *_batch methods will process their "
                          "inputs one at a time", stacklevel=2)

    def phoneticize_batch(self, texts: list[str]) -> list[str]:
        found = [(t, m.start(), m.end()) for t, text in enumerate(texts) for m in self.pattern.finditer(text)]
        readings = self._read([sentence_item(texts[t], start, end) for t, start, end in found])
        out = list(texts)
        for (t, start, end), reading in reversed(list(zip(found, readings))):  # from the end, so offsets stay valid
            out[t] = out[t][:start] + reading + out[t][end:]
        return out

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        items = []
        for sentence, word in pairs:
            self._check(word)
            match = word_pattern([word]).search(sentence)
            if not match:
                raise ValueError(f"{word!r} does not occur as a whole word in {sentence!r}")
            items.append(sentence_item(sentence, match.start(), match.end()))
        return self._read(items)

    def predict_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        items = []
        for text in texts:
            plain, spans = self._unmark(text, marks)
            items.append(sentence_item(plain, *spans[0]))  # the first marked word
        return self._read(items)

    def phoneticize_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        found = []  # (text index, start, end) of every marked word, in the text without marks
        out = []
        for t, text in enumerate(texts):
            plain, spans = self._unmark(text, marks)
            out.append(plain)
            found += [(t, start, end) for start, end in spans]
        readings = self._read([sentence_item(out[t], start, end) for t, start, end in found])
        for (t, start, end), reading in reversed(list(zip(found, readings))):
            out[t] = out[t][:start] + reading + out[t][end:]
        return out

    def _unmark(self, text: str, marks: tuple[str, str]) -> tuple[str, list[tuple[int, int]]]:
        """The text without marks and the span of each marked word in it; every marked word must be a heteronym."""
        plain, spans, pos = "", [], 0
        for m in re.finditer(re.escape(marks[0]) + "(.*?)" + re.escape(marks[1]), text):
            self._check(m.group(1))
            plain += text[pos:m.start()]
            spans.append((len(plain), len(plain) + len(m.group(1))))
            plain += m.group(1)
            pos = m.end()
        if not spans:
            raise ValueError(f"mark the heteronym as {marks[0]}word{marks[1]}: {text!r}")
        return plain + text[pos:], spans

    def _check(self, word: str) -> None:
        if word not in self.heteronyms:
            raise ValueError(f"{word!r} is not one of the {len(self.heteronyms)} known heteronyms")

    def _read(self, items: list[tuple[str, int, int, str]]) -> list[str]:
        """The reading for each (sentence, start, end, word); one model call per batch_size items (1 = sequential)."""
        # Every public method ends here. Each item is one sentence and one word in it, e.g.
        #   ("সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার কর।", 48, 50, "কর")
        # and each comes back as that word's reading, e.g. "করো".
        readings = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            with torch.inference_mode():
                if self.config["model_type"] == "seq2seq":
                    readings += self._read_seq2seq(chunk)
                else:
                    readings += self._read_encoder(chunk)
        return readings

    def _read_encoder(self, items) -> list[str]:
        """Encoder classifier and candidate scorer."""
        batch = encoder_batch(self.tokenizer, items, self.config, self.norm, self.tokenizer.pad_token_id)
        # MODEL CALL. Input, one row per item (tokens as the BanglaBERT tokenizer prints them):
        #   encoder:   [CLS] কর [SEP] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP]
        #   candidate: one row per reading of the word, all scored in this same call:
        #              [CLS] সরকার কর বাড়িয়েছে , তাই তুমি এখনই হিসাব ##টা আবার @ কর @ । [SEP] কর্ [SEP]
        #              [CLS] ... আবার @ কর @ । [SEP] করো [SEP]
        #              [CLS] ... আবার @ কর @ । [SEP] কোরো [SEP]
        # Output: scores of shape (items, 3), one per reading slot (slots the word does not have are masked), e.g.
        #   encoder: [-4.36, 3.86, 0.37] = softmax 0.0003 / 0.9699 / 0.0298 for কর্‌ / করো / কোরো -> argmax 1 -> করো
        #   candidate: softmax 0.0001 / 0.9995 / 0.0005 -> argmax 1 -> করো
        scores = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        return [self.heteronyms[w][k] for (*_, w), k in zip(items, scores.argmax(-1).tolist())]

    def _read_seq2seq(self, items) -> list[str]:
        """Seq2seq (BanglaT5, mT5)."""
        inputs = ["phonetic: " + mark(s, a, b, self.config["marker"], self.norm)[0] for s, a, b, _ in items]
        batch = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.config["max_length"], return_tensors="pt")
        # MODEL CALL. Input, one text per item:  phonetic: সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার @ কর @।
        # Output, one text per item (greedy):    সরকার কর বাড়িয়েছে, তাই তুমি এখনই হিসাবটা আবার করো।
        # The generated word at the target's position (করো) is matched to the word's readings below.
        output = self.model.generate(**batch.to(self.device), max_new_tokens=self.config["max_length"], num_beams=1, do_sample=False)
        readings = []
        for text, (sentence, start, _, word) in zip(self.tokenizer.batch_decode(output, skip_special_tokens=True), items):
            if self.config["output"] == "target":  # target-only checkpoint: the output is just the reading
                generated = strip_punct(text)
            else:
                token, piece = target_piece(sentence, start)
                tokens = text.split()
                pieces = re.split(rf"[{BOUNDARY}]", tokens[token]) if token < len(tokens) else []
                generated = strip_punct(pieces[piece]) if piece < len(pieces) else ""
            # compare after the same normalizer + tokenizer round trip, and return the candidate (it keeps its ZWNJ)
            canon = [self._roundtrip(c) for c in self.heteronyms[word]]
            readings.append(self.heteronyms[word][canon.index(generated)] if generated in canon else generated or word)
        return readings

    def _roundtrip(self, text: str) -> str:
        return strip_punct(self.tokenizer.decode(self.tokenizer(self.norm(text))["input_ids"], skip_special_tokens=True))
