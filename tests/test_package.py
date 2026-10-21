import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import bangla_heteronym_maker as bhm
from bangla_heteronym_maker import HeteronymModelReader, HeteronymReader, Reading
from bangla_heteronym_maker.text import word_pattern
from src.predict import Predictor

CANDIDATES = {
    "আঁক": ["আঁক্‌", "আঁকো"],
    "কর": ["কর্‌", "করো", "কোরো"],
    "বল": ["বল্‌", "বলো"],
}


class FakeReader(HeteronymModelReader):
    """The package's text handling with a model that always answers the first candidate."""

    def __init__(self, batch_size: int = 64):
        self.candidates = CANDIDATES
        self.pattern = word_pattern(list(CANDIDATES))
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.calls: list[list[tuple]] = []

    def _predict_items(self, items):
        self.calls.append(items)
        for sentence, start, end, word in items:
            assert sentence[start:end] == word
        return [(self.candidates[w][0], 1.0) for *_, w in items]


def first(word: str) -> str:
    return CANDIDATES[word][0]


def test_implements_the_interface():
    assert isinstance(FakeReader(), HeteronymReader)


def test_batch_methods_send_all_inputs_in_one_call():
    reader = FakeReader()
    texts = ["তুমি একটু অপেক্ষা কর, আমি আসছি। বল নিয়ে এসো।", "কোনো শব্দ নেই।", "আগে কর কাজটা, তারপর আবার কর।"]
    out = reader.phoneticize_batch(texts)
    assert len(reader.calls) == 1 and len(reader.calls[0]) == 4
    assert out[0] == f"তুমি একটু অপেক্ষা {first('কর')}, আমি আসছি। {first('বল')} নিয়ে এসো।"
    assert out[1] == texts[1]
    assert out[2] == f"আগে {first('কর')} কাজটা, তারপর আবার {first('কর')}।"
    assert reader.phoneticize(texts[0]) == out[0]

    reader.calls.clear()
    assert reader.predict_batch([("সময়মতো কর না দিলে", "কর"), ("বল নিয়ে এসো", "বল")]) == [first("কর"), first("বল")]
    assert reader.predict_marked_batch(["প্রথম বাক্য। আবার [কর]।", "তুমি [বল] নিয়ে এসো।"]) == [first("কর"), first("বল")]
    assert [len(c) for c in reader.calls] == [2, 2]


def test_readings_positions_and_marked_occurrence():
    reader = FakeReader()
    text = "আগে কর কাজটা, তারপর আবার কর।"
    readings = reader.readings(text)
    assert [(r.word, text[r.start:r.end]) for r in readings] == [("কর", "কর"), ("কর", "কর")]
    assert all(isinstance(r, Reading) for r in readings)
    reader.predict_marked("প্রথম বাক্য। আগে কর কাজটা, তারপর আবার [কর]।")
    sentence, start, _, _ = reader.calls[-1][0]
    assert sentence == " আগে কর কাজটা, তারপর আবার কর।" and start == sentence.rindex("কর")


def test_errors():
    reader = FakeReader()
    for bad in (lambda: reader.predict_marked("কোনো চিহ্ন নেই"), lambda: reader.predict_marked("এটা [তালিকায়] নেই"),
                lambda: reader.predict("এখানে কিছু নেই", "কর"), lambda: reader.predict("কর দাও", "তালিকায়")):
        with pytest.raises(ValueError):
            bad()


def test_singleton(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(HeteronymModelReader, "load", classmethod(lambda cls, path, *a: loads.append(path) or FakeReader()))
    bhm.reset()
    with pytest.raises(RuntimeError):
        bhm.get_reader()
    first_reader = bhm.initialize(tmp_path)
    assert bhm.initialize(str(tmp_path)) is first_reader is bhm.get_reader() and len(loads) == 1
    with pytest.raises(RuntimeError):
        bhm.initialize(tmp_path / "other")
    bhm.reset()
    monkeypatch.setenv(bhm.MODEL_PATH_ENV, str(tmp_path))
    assert bhm.get_reader() is not first_reader and bhm.is_initialized() and len(loads) == 2
    bhm.reset()


def test_package_is_self_contained(tmp_path):
    """The folder works after being copied elsewhere: no import from this repo's src/."""
    shutil.copytree("bangla_heteronym_maker", tmp_path / "bangla_heteronym_maker")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = "import bangla_heteronym_maker as b, sys; assert not any(m == 'src' or m.startswith('src.') for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


TEXTS = ["আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।",
         "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার কর।", "আজ ছুটি।"]


@pytest.mark.parametrize("name", ["banglabert", "banglabert_cand", "banglat5"])
def test_same_output_as_benchmark_predictor(name):
    path = Path("checkpoints") / name
    if not path.exists():
        pytest.skip("no trained checkpoint")
    reader = HeteronymModelReader.load(path.resolve(), device="cpu")
    predictor = Predictor.load(path, device="cpu")
    detailed = predictor.phoneticize_detailed_batch(TEXTS)
    assert reader.phoneticize_batch(TEXTS) == [t for t, _ in detailed] == predictor.phoneticize_batch(TEXTS)
    assert [[(r.word, r.start, r.reading) for r in rs] for rs in reader.readings_batch(TEXTS)] == \
        [[(d["word"], d["position"], d["prediction"]) for d in ds] for _, ds in detailed]
    assert reader.phoneticize_batch(TEXTS) == [reader.phoneticize(t) for t in TEXTS]
    pairs = [("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল")]
    assert reader.predict_batch(pairs) == predictor.predict_batch(pairs) == [predictor.predict(*p) for p in pairs]
    marked = ["সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।", "তুমি গ্রাফ পেপারে সরলরেখাটা [আঁক]।"]
    assert reader.predict_marked_batch(marked) == predictor.predict_marked_batch(marked) == [reader.predict_marked(m) for m in marked]
