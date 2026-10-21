import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import bangla_heteronym_package as bhp
from bangla_heteronym_package import HeteronymModelReader, HeteronymReader
from bangla_heteronym_package.text import word_pattern
from src.predict import Predictor

CANDIDATES = {"আঁক": ["আঁক্‌", "আঁকো"], "কর": ["কর্‌", "করো", "কোরো"], "বল": ["বল্‌", "বলো"]}


class FakeReader(HeteronymModelReader):
    """The package's text handling with a model that always answers the first reading."""

    def __init__(self, *args):
        self.heteronyms = CANDIDATES
        self.pattern = word_pattern(CANDIDATES)
        self.calls = []

    def _read(self, items):
        self.calls.append(items)
        assert all(sentence[start:end] == word for sentence, start, end, word in items)
        return [self.heteronyms[w][0] for *_, w in items]


def first(word: str) -> str:
    return CANDIDATES[word][0]


def test_batch_methods_send_all_inputs_in_one_call():
    reader = FakeReader()
    assert isinstance(reader, HeteronymReader)
    texts = ["তুমি একটু অপেক্ষা কর, আমি আসছি। বল নিয়ে এসো।", "কোনো শব্দ নেই।", "আগে কর কাজটা, তারপর আবার কর।"]
    out = reader.phoneticize_batch(texts)
    assert len(reader.calls) == 1 and len(reader.calls[0]) == 4
    assert out == [f"তুমি একটু অপেক্ষা {first('কর')}, আমি আসছি। {first('বল')} নিয়ে এসো।", texts[1],
                   f"আগে {first('কর')} কাজটা, তারপর আবার {first('কর')}।"]
    assert reader.phoneticize(texts[0]) == out[0]
    reader.calls.clear()
    assert reader.predict_batch([("সময়মতো কর না দিলে", "কর"), ("বল নিয়ে এসো", "বল")]) == [first("কর"), first("বল")]
    assert reader.predict_marked_batch(["প্রথম বাক্য। আবার [কর]।"]) == [first("কর")]
    assert [len(c) for c in reader.calls] == [2, 1]


def test_marked_occurrence_is_read_in_its_own_sentence():
    reader = FakeReader()
    assert reader.predict_marked("প্রথম বাক্য। আগে কর কাজটা, তারপর আবার [কর]।") == first("কর")
    sentence, start, _, _ = reader.calls[-1][0]
    assert sentence == " আগে কর কাজটা, তারপর আবার কর।" and start == sentence.rindex("কর")
    assert reader.predict_marked("তুমি <<বল>> আনো।", marks=("<<", ">>")) == first("বল")


def test_phoneticize_marked_replaces_only_marked_words_and_removes_marks():
    reader = FakeReader()
    texts = ["আগে কর কাজটা, তারপর আবার [কর]। [বল] আনো।", "তুমি <<আঁক>>।"]
    assert reader.phoneticize_marked_batch(texts[:1]) == [f"আগে কর কাজটা, তারপর আবার {first('কর')}। {first('বল')} আনো।"]
    assert len(reader.calls) == 1 and [i[0] for i in reader.calls[0]] == ["আগে কর কাজটা, তারপর আবার কর।", " বল আনো।"]
    assert reader.phoneticize_marked(texts[1], marks=("<<", ">>")) == f"তুমি {first('আঁক')}।"
    with pytest.raises(ValueError):
        reader.phoneticize_marked("আবার কর।")


def test_errors():
    reader = FakeReader()
    for bad in (lambda: reader.predict_marked("কোনো চিহ্ন নেই"), lambda: reader.predict_marked("এটা [তালিকায়] নেই"),
                lambda: reader.predict("এখানে কিছু নেই", "কর"), lambda: reader.predict("কর দাও", "তালিকায়")):
        with pytest.raises(ValueError):
            bad()


def test_singleton_and_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(bhp, "HeteronymModelReader", FakeReader)
    monkeypatch.delenv("BANGLA_HETERONYM_MODEL", raising=False)
    bhp.reset()
    with pytest.raises(ValueError):
        bhp.initialize()
    reader = bhp.initialize(tmp_path)
    assert bhp.initialize(str(tmp_path)) is reader is bhp.get_reader()
    with pytest.raises(RuntimeError):
        bhp.initialize(tmp_path / "other")
    bhp.reset()
    monkeypatch.setenv("BANGLA_HETERONYM_MODEL", str(tmp_path))
    assert bhp.get_reader() is not reader
    bhp.reset()


def test_env_file(monkeypatch, tmp_path):
    shutil.copytree("bangla_heteronym_package", tmp_path / "bangla_heteronym_package", ignore=shutil.ignore_patterns(".env", "__pycache__"))
    (tmp_path / "bangla_heteronym_package" / ".env").write_text("BANGLA_HETERONYM_MODEL=/models/x\nBANGLA_HETERONYM_DEVICE='cpu'\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "BANGLA_HETERONYM_MODEL", "BANGLA_HETERONYM_DEVICE")}
    code = ("import sys, bangla_heteronym_package as b; "
            "assert not any(m == 'src' or m.startswith('src.') for m in sys.modules); "  # self-contained
            "print(b._setting('BANGLA_HETERONYM_MODEL'), b._setting('BANGLA_HETERONYM_DEVICE'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert out.stdout.split() == ["/models/x", "cpu"]


TEXTS = ["আগে কর কাজটা, তারপর আবার কর। বিকেলে মাঠে বল নিয়ে যাব।", "তুমি গ্রাফ পেপারে সরলরেখাটা আঁক।",
         "সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার কর।", "আজ ছুটি।"]


@pytest.mark.parametrize("name", ["banglabert", "banglabert_cand", "banglat5"])
def test_same_output_as_benchmark_predictor(name):
    path = Path("checkpoints") / name
    if not path.exists():
        pytest.skip("no trained checkpoint")
    reader = HeteronymModelReader(path.resolve(), device="cpu")
    predictor = Predictor.load(path, device="cpu")
    assert reader.phoneticize_batch(TEXTS) == predictor.phoneticize_batch(TEXTS) == [reader.phoneticize(t) for t in TEXTS]
    pairs = [("সময়মতো কর না দিলে জরিমানা গুনতে হবে।", "কর"), ("বিকেলে মাঠে বল নিয়ে যাব।", "বল")]
    assert reader.predict_batch(pairs) == predictor.predict_batch(pairs)
    marked = ["সরকার কর বাড়িয়েছে, তাই হিসাবটা আবার [কর]।", "তুমি গ্রাফ পেপারে সরলরেখাটা [আঁক]।"]
    assert reader.predict_marked_batch(marked) == predictor.predict_marked_batch(marked) == [reader.predict_marked(m) for m in marked]
    marked_texts = ["সরকার [কর] বাড়িয়েছে, তাই হিসাবটা আবার [কর]। বিকেলে মাঠে [বল] নিয়ে যাব।", "তুমি গ্রাফ পেপারে সরলরেখাটা [আঁক]।"]
    assert reader.phoneticize_marked_batch(marked_texts) == predictor.phoneticize_marked_batch(marked_texts)


def test_no_batch_support_warns_and_runs_one_at_a_time(monkeypatch):
    import torch
    from bangla_heteronym_package import reader as reader_module

    class CountingModel(torch.nn.Module):
        calls = []

        def forward(self, rows):
            self.calls.append(len(rows))
            return torch.zeros(len(rows), 3)

    config = {"model_type": "encoder", "batch": False, "precision": "fp32", "candidates": CANDIDATES, "normalize": False}
    monkeypatch.setattr(reader_module, "load_checkpoint", lambda path: (CountingModel(), type("Tokenizer", (), {"pad_token_id": 0})(), config))
    monkeypatch.setattr(reader_module, "encoder_batch", lambda tok, items, *a: {"rows": torch.zeros(len(items))})
    with pytest.warns(UserWarning, match="does not support batch inference"):
        reader = reader_module.HeteronymModelReader("/models/x", device="cpu")
    assert reader.phoneticize_batch(["কর দাও, বল আনো।", "আবার কর।"]) == [f"{first('কর')} দাও, {first('বল')} আনো।", f"আবার {first('কর')}।"]
    assert CountingModel.calls == [1, 1, 1]
