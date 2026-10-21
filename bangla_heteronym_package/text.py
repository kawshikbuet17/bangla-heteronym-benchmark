import re
import string

ZWNJ = "‌"  # readings like কর্‌ end in hasanta + ZWNJ; keep it in outputs
PUNCT = "।॥" + string.punctuation + "“”‘’—–"  # word boundaries besides whitespace (\b breaks Bangla words)
BOUNDARY = re.escape(PUNCT)
SENTENCE_END = re.compile(r"(?<=[।?!\n])")
CONTEXT_WORDS = 32  # words kept on each side of the target in very long sentences


def get_normalizer(enabled: bool, strip_zwnj: bool):
    norm = lambda text: text
    if enabled:
        from normalizer import normalize as norm  # BanglaBERT / BanglaT5 checkpoints only
    return (lambda text: norm(text).replace(ZWNJ, "")) if strip_zwnj else norm


def word_pattern(words) -> re.Pattern:
    """Matches any of the words as a whole word."""
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![^\s{BOUNDARY}])(?:{alternation})(?![^\s{BOUNDARY}])")


def strip_punct(token: str) -> str:
    return token.strip(PUNCT + " \t\n")


def target_piece(sentence: str, start: int) -> tuple[int, int]:
    """(whitespace token index, piece index inside that token split on punctuation) of the word at `start`."""
    before = sentence[:start]
    token = len(before.split()) - (1 if before and not before[-1].isspace() else 0)
    piece = len(re.split(rf"[{BOUNDARY}]", re.split(r"\s", before)[-1])) - 1
    return token, piece


def mark(sentence: str, start: int, end: int, marker, norm) -> tuple[str, int, int, int]:
    """Puts the marker around one occurrence. Returns the text, the anchor char (opening marker) and the word's span."""
    left, word, right = norm(sentence[:start]), norm(sentence[start:end]), norm(sentence[end:])
    opening, closing = (marker[0] + " ", " " + marker[1]) if marker else ("", "")
    word_start = len(left) + len(opening)
    return left + opening + word + closing + right, len(left), word_start, word_start + len(word)


def sentence_item(text: str, start: int, end: int) -> tuple[str, int, int, str]:
    """(sentence, start, end, word) for the word at text[start:end], cut to its sentence and to CONTEXT_WORDS."""
    offset = 0
    for sentence in SENTENCE_END.split(text):
        if offset <= start < offset + len(sentence):
            s, e = start - offset, end - offset
            left, right = sentence[:s].split(" "), sentence[e:].split(" ")
            if len(left) > CONTEXT_WORDS or len(right) > CONTEXT_WORDS:
                left = " ".join(left[-CONTEXT_WORDS:])
                sentence, s, e = left + sentence[s:e] + " ".join(right[:CONTEXT_WORDS]), len(left), len(left) + e - s
            return sentence, s, e, sentence[s:e]
        offset += len(sentence)
    raise ValueError(f"position {start} is outside the text")
