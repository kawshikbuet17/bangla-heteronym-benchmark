import pytest

from src.data import RAW_DIR, build_input, find_occurrences, piece_index, token_index, word_pattern


def test_whole_word_matching_without_word_boundary_regex():
    pattern = word_pattern(["আসল", "আসলে"])
    text = "আসলে ব্যাপারটা আসল, আসলটা নয়।আসল"
    assert [w for *_, w in find_occurrences(text, pattern)] == ["আসলে", "আসল", "আসল"]


def test_build_input_marks_one_occurrence():
    sentence = "তুমি এবার বল কী হয়েছিল।"
    start = sentence.index("বল")
    text, anchor, ts, te = build_input(sentence, start, start + 2, ("@", "@"), lambda s: s)
    assert text == "তুমি এবার @ বল @ কী হয়েছিল।"
    assert text[anchor] == "@" and text[ts:te] == "বল"
    assert build_input(sentence, start, start + 2, None, lambda s: s)[0] == sentence


def test_token_and_piece_index():
    sentence = "কালো আর সাদা-কালো “বল” দেখি"
    assert (token_index(sentence, sentence.index("আর")), piece_index(sentence, sentence.index("আর"))) == (1, 0)
    assert (token_index(sentence, sentence.rindex("কালো")), piece_index(sentence, sentence.rindex("কালো"))) == (2, 1)
    assert (token_index(sentence, sentence.index("বল")), piece_index(sentence, sentence.index("বল"))) == (3, 1)


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw data not downloaded (python -m src.data)")
def test_data_and_split_checks():
    from src.splits import check

    report = check()
    assert report["sizes"]["train"] == 5152 and report["sizes"]["test"] == 1104
