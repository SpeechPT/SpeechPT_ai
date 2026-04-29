from speechpt.coherence.transcript_aligner import align_transcript


def test_align_transcript_preserves_empty_slide_segments():
    words = [
        {"word": "첫", "start": 0.1, "end": 0.3},
        {"word": "슬라이드", "start": 0.4, "end": 0.8},
        {"word": "셋째", "start": 10.2, "end": 10.5},
    ]

    segments = align_transcript(words, [0.0, 5.0, 10.0, 15.0])

    assert len(segments) == 3
    assert segments[0].slide_id == 1
    assert segments[0].text == "첫 슬라이드"
    assert segments[1].slide_id == 2
    assert segments[1].text == ""
    assert "empty_segment" in segments[1].warning_flags
    assert segments[2].slide_id == 3
    assert segments[2].text == "셋째"


def test_align_transcript_returns_boundary_based_empty_segments_when_no_words():
    segments = align_transcript([], [0.0, 4.0, 9.0])

    assert len(segments) == 2
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 4.0
    assert segments[1].start_sec == 4.0
    assert segments[1].end_sec == 9.0
    assert all("empty_segment" in seg.warning_flags for seg in segments)
