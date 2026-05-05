import pytest

from speechpt.attitude.filler_detector import detect_fillers, is_filler


def test_is_filler_matches_korean_fillers_as_full_tokens():
    assert is_filler("어")
    assert is_filler("어어")
    assert is_filler("음")
    assert is_filler("그러니까")
    assert not is_filler("음악")
    assert not is_filler("그림")


def test_detect_fillers_counts_rate_and_per_slide():
    words = [
        {"word": "안녕하세요", "start": 0.0, "end": 0.4},
        {"word": "어어", "start": 1.0, "end": 1.2},
        {"word": "내용", "start": 5.0, "end": 5.3},
        {"word": "그러니까", "start": 12.0, "end": 12.4},
    ]

    result = detect_fillers(words, slide_timestamps=[0.0, 10.0, 20.0])

    assert result["total_words"] == 4
    assert result["total_fillers"] == 2
    assert result["filler_rate"] == 0.5
    assert [item["word"] for item in result["filler_words"]] == ["어어", "그러니까"]
    assert result["per_slide"][1]["count"] == 1
    assert result["per_slide"][1]["rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["per_slide"][2]["count"] == 1
    assert result["per_slide"][2]["rate"] == 1.0


def test_detect_fillers_accepts_additional_patterns():
    words = [{"word": "흠", "start": 0.0, "end": 0.2}]

    result = detect_fillers(words, patterns=[r"^흠$"])

    assert result["total_fillers"] == 1
    assert result["filler_words"][0]["word"] == "흠"
