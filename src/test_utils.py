import pytest
from src.utils import flatten, normalize_text, batch_is_empty, batch_count_substring, batch_count_lowercase


def test_flatten_basic():
    assert flatten([[1, 2], [3], 4]) == [1, 2, 3, 4]
    assert flatten([[], 1, 2]) == [1, 2]
    assert flatten([1, 2, 3]) == [1, 2, 3]
    assert flatten([["a", "b"], ["c"], "d"]) == ["a", "b", "c", "d"]

def test_flatten_nested_lists():
    assert flatten([[1, [2, 3]], [4], 5]) == [1, [2, 3], 4, 5]
    assert flatten([[[], []], [1], 2]) == [1, 2]

def test_normalize_text_basic():
    assert normalize_text("  HeLLo   WoRLd  ") == "hello world"
    assert normalize_text("\tTabs\nNewlines\t  ") == "tabs newlines"
    assert normalize_text("Word   Word") == "word word"
    assert normalize_text("") == ""

def test_batch_is_empty():
    batch = ["", "   ", "hello", None, "\n"]
    expected = [True, True, False, True, True]
    assert batch_is_empty(batch) == expected

def test_batch_count_substring():
    texts = ["banana", "bandana", "an", "", None]
    result = batch_count_substring(texts, "an")
    assert result == [2, 2, 1, 0, 0]


def test_batch_count_lowercase():
    texts = ["abcDEF", "ALLUP", "mixedCase", "", None]
    expected = [3, 0, 5, 0, 0]
    assert batch_count_lowercase(texts) == expected
