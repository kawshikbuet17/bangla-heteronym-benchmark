import os
from pathlib import Path

import pytest

from src.data import ZWNJ, word_pattern
from src.predict import Prediction, Predictor

CANDIDATES = {
    "আঁক": ["আঁক্‌", "আঁকো"],
    "কর": ["কর্‌", "করো", "কোরো"],
    "খেল": ["খ্যাল্‌", "খ্যালো", "খেলো"],
    "বল": ["বল্‌", "বলো"],
}


class FakePredictor(Predictor):
    """Same phoneticize logic, but the model always answers the first candidate."""

    def __init__(self):
        self.candidates = CANDIDATES
        self.pattern = word_pattern(list(CANDIDATES))
        self.calls: list[list[tuple]] = []

    def predict_items(self, items, batch_size=64):
        self.calls.append(items)
        for sentence, start, end, word in items:
            assert sentence[start:end] == word
        return [Prediction(self.candidates[w][0], 1.0) for *_, w in items]


def first(word: str) -> str:
    return CANDIDATES[word][0]


def test_multiple_sentences():
    predictor = FakePredictor()
    text = "তুমি একটু অপেক্ষা কর, আমি আসছি। বিকেলে পার্কে খেল!\nকাল আবার দেখা হবে?"
    expected = text.replace("কর,", first("কর") + ",").replace("খেল!", first("খেল") + "!")
    assert predictor.phoneticize(text) == expected
    assert len(predictor.calls) == 1 and len(predictor.calls[0]) == 2  # all occurrences batched


def test_two_different_heteronyms_in_one_sentence():
    text = "তুমি খাতায় আঁক আর পরে খেল।"
    assert FakePredictor().phoneticize(text) == f"তুমি খাতায় {first('আঁক')} আর পরে {first('খেল')}।"


def test_same_heteronym_twice():
    predictor = FakePredictor()
    text = "কর কাজটা, তারপর আবার কর।"
    assert predictor.phoneticize(text) == f"{first('কর')} কাজটা, তারপর আবার {first('কর')}।"
    _, details = predictor.phoneticize_detailed(text)
    assert [d["position"] for d in details] == [0, text.rindex("কর")]
    assert all(item[0].count("কর") == 2 for item in predictor.calls[-1])  # the full sentence is kept as context


def test_heteronym_next_to_punctuation():
    text = "সে বলল, “বল”—কেন বল? (বল) বল।"
    b = first("বল")
    assert FakePredictor().phoneticize(text) == f"সে বলল, “{b}”—কেন {b}? ({b}) {b}।"  # বলল is not a whole-word match


def test_no_heteronym_returns_text_unchanged():
    text = "আজ  আকাশ পরিষ্কার।\n\nআঁকা ছবিটা সুন্দর, করলাম না! "
    predictor = FakePredictor()
    assert predictor.phoneticize(text) == text
    assert predictor.calls == [[]]


def test_zwnj_preserved_in_output():
    out = FakePredictor().phoneticize("তুমি সরলরেখাটা আঁক।")
    assert out == "তুমি সরলরেখাটা আঁক্‌।"
    assert ZWNJ in out


CHECKPOINT = os.environ.get("HETERO_CHECKPOINT", "checkpoints/banglabert_small")


@pytest.mark.skipif(not Path(CHECKPOINT).exists(), reason="no trained checkpoint")
def test_real_checkpoint_keeps_structure_and_zwnj():
    predictor = Predictor.load(CHECKPOINT)
    text = "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক। ছোটবেলায় আঁক কষতে বসলেই আমার ঘুম পেত।\nআজ ছুটি।"
    out, details = predictor.phoneticize_detailed(text)
    assert len(details) == 2
    assert out.endswith("।\nআজ ছুটি।")
    assert all(d["prediction"] in predictor.candidates["আঁক"] for d in details)
    assert ZWNJ in out or all(ZWNJ not in d["prediction"] for d in details)


def test_predict_marked_uses_the_marked_occurrence_and_its_sentence():
    predictor = FakePredictor()
    assert predictor.predict_marked("প্রথম বাক্য। আগে কর কাজটা, তারপর আবার [কর]।") == first("কর")
    sentence, start, end, word = predictor.calls[-1][0]
    assert sentence == " আগে কর কাজটা, তারপর আবার কর।" and word == "কর" and start == sentence.rindex("কর")
    assert predictor.predict_marked("তুমি <<বল>> নিয়ে এসো।", marks=("<<", ">>")) == first("বল")
    with pytest.raises(ValueError):
        predictor.predict_marked("কোনো চিহ্ন নেই")
    with pytest.raises(ValueError):
        predictor.predict_marked("এটা [তালিকায়] নেই")


def test_batch_methods_make_one_predict_items_call():
    predictor = FakePredictor()
    texts = ["তুমি একটু অপেক্ষা কর, আমি আসছি।", "কোনো শব্দ নেই।", "বল নিয়ে এসো। আবার কর।"]
    out = predictor.phoneticize_batch(texts)
    assert len(predictor.calls) == 1 and len(predictor.calls[0]) == 3
    assert out == [predictor.phoneticize(t) for t in texts]
    predictor.calls.clear()
    assert predictor.predict_batch([("কর দাও", "কর"), ("বল নিয়ে এসো", "বল")]) == [first("কর"), first("বল")]
    assert predictor.predict_marked_batch(["আবার [কর]।", "[বল] আনো।"]) == [first("কর"), first("বল")]
    assert [len(c) for c in predictor.calls] == [2, 2]
