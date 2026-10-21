from abc import ABC, abstractmethod


class HeteronymReader(ABC):
    """Reads Bangla heteronyms in context. The *_batch methods send all inputs through the model together."""

    @abstractmethod
    def phoneticize_batch(self, texts: list[str]) -> list[str]:
        """Each text with every known heteronym replaced by its reading."""

    @abstractmethod
    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        """(sentence, word) pairs -> the reading of the word's first occurrence in its sentence."""

    @abstractmethod
    def predict_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        """Texts with one heteronym marked, e.g. "আবার [কর]।" -> the reading of the marked word."""

    @abstractmethod
    def phoneticize_marked_batch(self, texts: list[str], marks: tuple[str, str] = ("[", "]")) -> list[str]:
        """Each text with only its marked heteronyms replaced (and the marks removed), e.g. "আবার [কর]।" -> "আবার করো।"."""

    def phoneticize(self, text: str) -> str:
        return self.phoneticize_batch([text])[0]

    def predict(self, sentence: str, word: str) -> str:
        return self.predict_batch([(sentence, word)])[0]

    def predict_marked(self, text: str, marks: tuple[str, str] = ("[", "]")) -> str:
        return self.predict_marked_batch([text], marks)[0]

    def phoneticize_marked(self, text: str, marks: tuple[str, str] = ("[", "]")) -> str:
        return self.phoneticize_marked_batch([text], marks)[0]
