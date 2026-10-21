"""Text helpers: word boundaries, finding heteronyms, marking the target and splitting text into sentences."""
import re
import string
from collections.abc import Callable

ZWNJ = "‌"  # readings use hasanta + ZWNJ (U+09CD U+200C); never strip it from outputs
SEQ2SEQ_PREFIX = "phonetic: "

# Word boundaries: whitespace, danda / double danda, ASCII punctuation, curly quotes and dashes.
# `\b` cannot be used: it splits Bangla words at vowel signs.
BOUNDARY_CHARS = "।॥" + string.punctuation + "“”‘’—–"
_B = re.escape(BOUNDARY_CHARS)
_STRIP = BOUNDARY_CHARS + " \t\n"
SENTENCE_END = re.compile(r"(?<=[।?!\n])")
CONTEXT_WORDS = 32  # words kept on each side of the target when one "sentence" is very long

Item = tuple[str, int, int, str]  # sentence, target start, target end, word
Marker = tuple[str, str] | None
Normalizer = Callable[[str], str]


def get_normalizer(enabled: bool, strip_zwnj: bool = False) -> Normalizer:
    base = lambda text: text
    if enabled:
        from normalizer import normalize as base  # only BanglaBERT / BanglaT5 checkpoints need this package
    if not strip_zwnj:
        return base
    return lambda text: base(text).replace(ZWNJ, "")


def word_pattern(words: list[str]) -> re.Pattern:
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![^\s{_B}])(?:{alternation})(?![^\s{_B}])")


def find_occurrences(text: str, pattern: re.Pattern) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in pattern.finditer(text)]


def strip_punct(token: str) -> str:
    return token.strip(_STRIP)


def token_index(sentence: str, start: int) -> int:
    before = sentence[:start]
    n = len(before.split())
    return n - 1 if before and not before[-1].isspace() else n


def split_pieces(token: str) -> list[str]:
    return re.split(rf"[{_B}]", token)


def piece_index(sentence: str, start: int) -> int:
    """Index of the target inside its whitespace token split on punctuation (e.g. 1 for সাদা-কালো)."""
    token_prefix = re.split(r"\s", sentence[:start])[-1]
    return len(split_pieces(token_prefix)) - 1


def build_input(sentence: str, start: int, end: int, marker: Marker, norm: Normalizer) -> tuple[str, int, int, int]:
    """Marks one occurrence. Returns the text, the anchor char (opening marker, or the target if there is no
    marker) and the target's char span in the returned text."""
    left, target, right = norm(sentence[:start]), norm(sentence[start:end]), norm(sentence[end:])
    opening, closing = (marker[0] + " ", " " + marker[1]) if marker else ("", "")
    target_start = len(left) + len(opening)
    return left + opening + target + closing + right, len(left), target_start, target_start + len(target)


def crop(sentence: str, start: int, end: int) -> Item:
    """Keeps at most CONTEXT_WORDS words on each side of the target."""
    left, right = sentence[:start].split(" "), sentence[end:].split(" ")
    if len(left) <= CONTEXT_WORDS and len(right) <= CONTEXT_WORDS:
        return sentence, start, end, sentence[start:end]
    left = " ".join(left[-CONTEXT_WORDS:])
    return left + sentence[start:end] + " ".join(right[:CONTEXT_WORDS]), len(left), len(left) + end - start, sentence[start:end]


def sentence_item(text: str, start: int, end: int) -> Item:
    """The item for the occurrence text[start:end], cut to the sentence that contains it."""
    offset = 0
    for piece in SENTENCE_END.split(text):
        if offset <= start < offset + len(piece):
            return crop(piece, start - offset, end - offset)
        offset += len(piece)
    raise ValueError(f"position {start} is outside the text")
