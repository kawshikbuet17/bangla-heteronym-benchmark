"""The interface of a heteronym reader. `reader.HeteronymModelReader` implements it with a trained checkpoint."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    """One heteronym occurrence in the input text and how it is read."""
    word: str          # the heteronym as written, e.g. "কর"
    start: int         # character offsets of the word in the input text
    end: int
    reading: str       # its pronunciation as a Bangla respelling, e.g. "করো" (keeps the ZWNJ, e.g. "কর্‌")
    confidence: float  # 0..1


class HeteronymReader(ABC):
    """Reads Bangla heteronyms in context. Every *_batch method runs the whole list through the model in batches
    (one forward pass or one `generate` call per `batch_size` occurrences), not one call per input."""

    @property
    @abstractmethod
    def heteronyms(self) -> dict[str, list[str]]:
        """Every known heteronym and its readings, e.g. {"কর": ["কর্‌", "করো", "কোরো"], ...}."""

    # Whole text in, whole text out: every known heteronym is replaced by its reading.
    @abstractmethod
    def phoneticize_batch(self, texts: list[str]) -> list[str]: ...

    def phoneticize(self, text: str) -> str:
        return self.phoneticize_batch([text])[0]

    # Whole text in, one Reading per heteronym occurrence out.
    @abstractmethod
    def readings_batch(self, texts: list[str]) -> list[list[Reading]]: ...

    def readings(self, text: str) -> list[Reading]:
        return self.readings_batch([text])[0]

    # Sentence + word in, only the word's reading out (first whole-word occurrence of the word).
    @abstractmethod
    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[str]: ...

    def predict(self, sentence: str, word: str) -> str:
        return self.predict_batch([(sentence, word)])[0]

    # Text with the target marked, e.g. "আগে কর কাজটা, তারপর আবার [কর]।", in; only the marked word's reading out.
    @abstractmethod
    def predict_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]: ...

    def predict_marked(self, text: str, marks: tuple[str, str] = ("[", "]")) -> str:
        return self.predict_marked_batch([text], marks)[0]
