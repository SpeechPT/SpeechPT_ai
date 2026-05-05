import numpy as np

from speechpt.coherence.auto_aligner import (
    AlignmentResult,
    TimedUnit,
    _absorb_pause_chunks,
    _anchor_cover_and_thanks,
    _decode_flexible_monotonic_path,
    _refine_boundaries_with_title_cues,
    _resolve_cover_end,
    _smooth_low_confidence_runs,
    auto_align_slides,
    build_timed_units,
    is_short_title,
    normalize_manual_boundaries,
    resolve_alignment,
)
from speechpt.coherence.keypoint_extractor import Keypoint


class DummyModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            if "시장" in text or "성장" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "수익" in text or "비용" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "결론" in text or "마무리" in text or "감사" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.5, 0.5, 0.5])
        arr = np.array(vectors, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-8)
        return arr


def test_normalize_manual_boundaries_prepends_start_when_missing():
    words = [
        {"word": "도입", "start": 0.0, "end": 0.2},
        {"word": "결론", "start": 11.5, "end": 12.0},
    ]
    boundaries, warnings = normalize_manual_boundaries([4.0, 8.0, 12.0], slide_count=3, words=words)
    assert boundaries == [0.0, 4.0, 8.0, 12.0]
    assert "prepended_transcript_start" in warnings


def test_auto_align_slides_returns_slide_count_plus_one_boundaries(monkeypatch):
    monkeypatch.setattr("speechpt.coherence.auto_aligner._get_model", lambda _: DummyModel())
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.kss.split_sentences",
        lambda text: ["시장 성장입니다", "다음으로 수익 구조입니다"],
    )

    slide_keypoints = [
        [Keypoint(text="시장 규모", importance=1.0, source="title")],
        [Keypoint(text="수익 구조", importance=1.0, source="title")],
    ]
    words = [
        {"word": "시장", "start": 0.0, "end": 0.2},
        {"word": "성장입니다.", "start": 0.21, "end": 0.6},
        {"word": "다음으로", "start": 2.0, "end": 2.2},
        {"word": "수익", "start": 2.21, "end": 2.4},
        {"word": "구조입니다.", "start": 2.41, "end": 2.9},
    ]

    result = auto_align_slides(
        slide_keypoints=slide_keypoints,
        words=words,
        model_name="dummy",
        config={
            "pause_threshold_sec": 0.9,
            "chunk_min_duration_sec": 0.1,
            "min_first_chunk_sec": 0.0,
            "min_last_chunk_sec": 0.0,
            "anchor_cover_slide": False,
            "min_dwell_units": 1,
            "start_skip_penalty": 1.0,
        },
    )
    assert result.final_boundaries[0] == 0.0
    assert len(result.final_boundaries) == 3
    assert result.final_boundaries[-1] == 2.9
    assert result.unit_assignments[0]["slide_id"] == 1
    assert result.unit_assignments[-1]["slide_id"] == 2


def test_resolve_alignment_hybrid_keeps_manual_and_records_auto_proposal(monkeypatch):
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.auto_align_slides",
        lambda **kwargs: AlignmentResult(
            mode="auto",
            strategy_used="auto",
            final_boundaries=[0.0, 4.5, 9.0],
            proposed_boundaries=[0.0, 4.5, 9.0],
            confidence=0.84,
            unit_assignments=[{"unit_id": 0, "slide_id": 1}],
        ),
    )

    slide_keypoints = [
        [Keypoint(text="시장 규모", importance=1.0, source="title")],
        [Keypoint(text="수익 구조", importance=1.0, source="title")],
    ]
    words = [{"word": "시장", "start": 0.0, "end": 0.2}]

    result = resolve_alignment(
        slide_keypoints=slide_keypoints,
        words=words,
        model_name="dummy",
        mode="hybrid",
        provided_boundaries=[0.0, 5.0, 10.0],
        config={},
    )
    assert result.strategy_used == "manual_with_auto_proposal"
    assert result.final_boundaries == [0.0, 5.0, 10.0]
    assert result.proposed_boundaries == [0.0, 4.5, 9.0]
    assert result.confidence == 0.84


def test_build_timed_units_uses_kss_sentences_and_preserves_word_times(monkeypatch):
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.kss.split_sentences",
        lambda text: ["시장 규모를 설명드리겠습니다", "다음은 수익 구조입니다"],
    )

    words = [
        {"word": "시장", "start": 0.0, "end": 0.2},
        {"word": "규모를", "start": 0.21, "end": 0.45},
        {"word": "설명드리겠습니다", "start": 0.46, "end": 1.0},
        {"word": "다음은", "start": 2.0, "end": 2.3},
        {"word": "수익", "start": 2.31, "end": 2.55},
        {"word": "구조입니다", "start": 2.56, "end": 3.1},
    ]

    units = build_timed_units(
        words,
        config={
            "pause_threshold_sec": 0.9,
            "chunk_min_duration_sec": 0.1,
            "min_first_chunk_sec": 0.0,
            "min_last_chunk_sec": 0.0,
        },
    )

    assert len(units) == 2
    assert units[0].text == "시장 규모를 설명드리겠습니다"
    assert units[0].start_sec == 0.0
    assert units[0].end_sec == 1.0
    assert units[1].text == "다음은 수익 구조입니다"
    assert units[1].start_sec == 2.0
    assert units[1].end_sec == 3.1


def test_build_timed_units_force_merges_short_edge_chunks(monkeypatch):
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.kss.split_sentences",
        lambda text: ["안녕하세요", "시장 설명입니다", "감사합니다"],
    )
    words = [
        {"word": "안녕하세요", "start": 0.0, "end": 1.0},
        {"word": "시장", "start": 2.0, "end": 2.4},
        {"word": "설명입니다", "start": 2.41, "end": 5.0},
        {"word": "감사합니다", "start": 7.0, "end": 7.8},
    ]

    units = build_timed_units(
        words,
        config={
            "pause_threshold_sec": 0.5,
            "chunk_min_duration_sec": 0.1,
            "chunk_min_words": 1,
            "min_first_chunk_sec": 2.0,
            "min_last_chunk_sec": 2.0,
        },
    )

    assert len(units) == 1
    assert units[0].start_sec == 0.0
    assert units[0].end_sec == 7.8
    assert "안녕하세요" in units[0].text
    assert "감사합니다" in units[0].text


def test_is_short_title_raises_title_weight_condition():
    assert is_short_title("팀 소개")
    assert is_short_title("ControlV")
    assert not is_short_title("연속 제어 기반 오디오 스타일 변환 실험 결과")


def test_auto_align_slides_allows_trailing_slide_to_remain_empty(monkeypatch):
    monkeypatch.setattr("speechpt.coherence.auto_aligner._get_model", lambda _: DummyModel())
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.kss.split_sentences",
        lambda text: ["시장", "수익", "수익"],
    )
    slide_keypoints = [
        [Keypoint(text="시장 규모", importance=1.0, source="title")],
        [Keypoint(text="수익 구조", importance=1.0, source="title")],
        [Keypoint(text="결론", importance=1.0, source="title")],
    ]
    words = [
        {"word": "시장", "start": 0.0, "end": 0.2},
        {"word": "수익", "start": 2.0, "end": 2.2},
        {"word": "수익", "start": 4.0, "end": 4.2},
    ]

    result = auto_align_slides(
        slide_keypoints=slide_keypoints,
        words=words,
        model_name="dummy",
        config={
            "pause_threshold_sec": 0.9,
            "chunk_min_duration_sec": 0.1,
            "chunk_min_words": 1,
            "min_first_chunk_sec": 0.0,
            "min_last_chunk_sec": 0.0,
            "anchor_cover_slide": False,
            "min_dwell_units": 1,
            "start_skip_penalty": 1.0,
        },
    )

    assigned = [item["slide_id"] for item in result.unit_assignments]
    assert assigned == sorted(assigned)
    assert assigned[-1] == 2
    assert result.final_boundaries == [0.0, 2.0, 4.2, 4.2]


def test_auto_align_cover_anchor_keeps_opening_units_on_first_slide(monkeypatch):
    monkeypatch.setattr("speechpt.coherence.auto_aligner._get_model", lambda _: DummyModel())
    monkeypatch.setattr(
        "speechpt.coherence.auto_aligner.kss.split_sentences",
        lambda text: ["시장 발표를 시작하겠습니다", "수익 구조를 설명합니다"],
    )
    slide_keypoints = [
        [Keypoint(text="시장 발표", importance=1.0, source="title")],
        [Keypoint(text="수익 구조", importance=1.0, source="title")],
    ]
    words = [
        {"word": "시장", "start": 0.0, "end": 0.2},
        {"word": "발표를", "start": 0.3, "end": 1.0},
        {"word": "시작하겠습니다", "start": 1.1, "end": 10.0},
        {"word": "수익", "start": 12.0, "end": 12.3},
        {"word": "구조를", "start": 12.4, "end": 12.8},
        {"word": "설명합니다", "start": 12.9, "end": 20.0},
    ]

    result = auto_align_slides(
        slide_keypoints=slide_keypoints,
        words=words,
        model_name="dummy",
        config={
            "pause_threshold_sec": 0.9,
            "chunk_min_duration_sec": 0.1,
            "chunk_min_words": 1,
            "cover_min_window": 8.0,
            "cover_max_window": 16.0,
            "cover_ratio": 1.0,
        },
    )

    assert result.final_boundaries[1] == 12.0
    assert result.unit_assignments[0]["slide_id"] == 1


def test_cover_anchor_does_not_extend_past_window():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=10.0, text="표지", words=[]),
        TimedUnit(unit_id=1, start_sec=12.0, end_sec=26.0, text="팀 소개", words=[]),
    ]

    cover_end = _resolve_cover_end(
        units,
        config={
            "cover_min_window": 8.0,
            "cover_max_window": 20.0,
            "cover_ratio": 1.0,
        },
    )

    assert cover_end == 10.0


def test_cover_anchor_falls_back_to_window_when_no_unit_ends_inside():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=12.0, text="긴 도입", words=[]),
        TimedUnit(unit_id=1, start_sec=13.0, end_sec=20.0, text="본문", words=[]),
    ]

    cover_end = _resolve_cover_end(
        units,
        config={
            "cover_min_window": 8.0,
            "cover_max_window": 20.0,
            "cover_ratio": 0.1,
        },
    )

    assert cover_end == 8.0


def test_cover_anchor_stops_before_next_slide_title_cue():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=10.0, text="발표를 시작하겠습니다", words=[]),
        TimedUnit(unit_id=1, start_sec=11.0, end_sec=17.0, text="먼저 팀 소개 드리겠습니다", words=[]),
    ]
    slides = [
        [Keypoint(text="표지", importance=1.0, source="title")],
        [Keypoint(text="팀소개", importance=1.0, source="title")],
    ]

    cover_end = _resolve_cover_end(
        units,
        slides,
        config={
            "cover_min_window": 8.0,
            "cover_max_window": 20.0,
            "cover_ratio": 1.0,
        },
    )

    assert cover_end == 10.0


def test_dwell_decoder_prevents_single_unit_middle_slide():
    sim_matrix = np.array(
        [
            [1.0, 0.1, 0.1],
            [0.2, 0.95, 0.1],
            [0.1, 0.2, 0.95],
            [0.1, 0.2, 0.9],
        ],
        dtype=float,
    )

    assignments = _decode_flexible_monotonic_path(
        sim_matrix,
        config={
            "min_dwell_units": 2,
            "allow_skip": True,
            "progress_penalty": 0.0,
            "start_skip_penalty": 0.2,
        },
    )

    runs = []
    for slide_id in assignments:
        if not runs or runs[-1][0] != slide_id:
            runs.append([slide_id, 1])
        else:
            runs[-1][1] += 1
    assert all(length >= 2 for _, length in runs)


def test_thanks_anchor_ignores_early_closing_cue():
    units = [
        type("Unit", (), {"start_sec": 0.0, "end_sec": 3.0, "text": "들어주셔서 가능한 구조입니다"})(),
        type("Unit", (), {"start_sec": 90.0, "end_sec": 95.0, "text": "실험 결과입니다"})(),
        type("Unit", (), {"start_sec": 101.0, "end_sec": 104.0, "text": "감사합니다"})(),
    ]
    slides = [
        [Keypoint(text="표지", importance=1.0, source="title")],
        [Keypoint(text="결과", importance=1.0, source="title")],
        [Keypoint(text="감사합니다", importance=1.0, source="title")],
    ]

    anchors = _anchor_cover_and_thanks(units, slides, {"thanks_search_start_ratio": 0.85})
    assert anchors.thanks_start_sec == 101.0

    early_only = _anchor_cover_and_thanks(units[:2], slides, {"thanks_search_start_ratio": 0.85})
    assert early_only.thanks_start_sec == 95.0
    assert early_only.thanks_anchored


def test_confidence_smoothing_runs_once_absorbs_short_low_margin_run():
    assignments = [0, 1, 2, 2]
    sim_matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.31, 0.32],
            [0.0, 0.2, 0.9],
            [0.0, 0.1, 0.8],
        ],
        dtype=float,
    )

    smoothed = _smooth_low_confidence_runs(
        assignments,
        sim_matrix,
        config={
            "smoothing_margin_threshold": 0.1,
            "smoothing_max_units": 1,
            "smoothing_preserve_assigned_slides": False,
        },
        fixed_slides={0},
    )

    assert smoothed == [0, 2, 2, 2]


def test_absorb_pause_chunks_moves_short_weak_run_to_better_neighbor():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=8.0, text="도입", words=[]),
        TimedUnit(unit_id=1, start_sec=8.5, end_sec=10.0, text="약한 호흡 chunk", words=[]),
        TimedUnit(unit_id=2, start_sec=11.0, end_sec=20.0, text="다음", words=[]),
    ]
    assignments = [0, 1, 2]
    sim_matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.62, 0.4, 0.2],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )

    absorbed = _absorb_pause_chunks(
        assignments,
        units,
        sim_matrix,
        config={
            "absorb_min_chunk_sec": 3.0,
            "absorb_edge_max_sec": 9.0,
            "absorb_max_chunk_utts": 2,
            "absorb_margin_threshold": 0.05,
        },
    )

    assert absorbed == [0, 0, 2]


def test_absorb_pause_chunks_keeps_strong_run():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=8.0, text="도입", words=[]),
        TimedUnit(unit_id=1, start_sec=8.5, end_sec=10.0, text="강한 chunk", words=[]),
        TimedUnit(unit_id=2, start_sec=11.0, end_sec=20.0, text="다음", words=[]),
    ]
    assignments = [0, 1, 2]
    sim_matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )

    absorbed = _absorb_pause_chunks(
        assignments,
        units,
        sim_matrix,
        config={"absorb_margin_threshold": 0.05},
    )

    assert absorbed == assignments


def test_absorb_pause_chunks_excludes_cover_and_thanks_segments():
    units = [
        TimedUnit(unit_id=0, start_sec=0.0, end_sec=8.0, text="표지", words=[]),
        TimedUnit(unit_id=1, start_sec=8.5, end_sec=10.0, text="약한 chunk", words=[]),
        TimedUnit(unit_id=2, start_sec=11.0, end_sec=20.0, text="감사", words=[]),
    ]
    assignments = [0, 1, 2]
    sim_matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.62, 0.4, 0.7],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )

    absorbed = _absorb_pause_chunks(
        assignments,
        units,
        sim_matrix,
        config={
            "absorb_min_chunk_sec": 3.0,
            "absorb_edge_max_sec": 9.0,
            "absorb_max_chunk_utts": 2,
            "absorb_margin_threshold": 0.05,
        },
        fixed_slides={0, 2},
    )

    assert absorbed == assignments


def test_absorb_pause_chunks_moves_weak_leading_edge_to_previous_run():
    units = [
        TimedUnit(unit_id=0, start_sec=124.0, end_sec=164.0, text="해결 방법", words=[]),
        TimedUnit(unit_id=1, start_sec=164.1, end_sec=172.1, text="구간별 변화점", words=[]),
        TimedUnit(unit_id=2, start_sec=174.2, end_sec=217.0, text="현재까지 진행 현황", words=[]),
    ]
    assignments = [4, 5, 5]
    sim_matrix = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.8, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.55, 0.5],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.75],
        ],
        dtype=float,
    )

    absorbed = _absorb_pause_chunks(
        assignments,
        units,
        sim_matrix,
        config={
            "absorb_edge_max_sec": 9.0,
            "absorb_max_chunk_utts": 2,
            "absorb_margin_threshold": 0.05,
        },
    )

    assert absorbed == [4, 4, 5]


def test_title_cue_refinement_moves_boundary_inside_mixed_unit():
    units = [
        TimedUnit(
            unit_id=0,
            start_sec=154.0,
            end_sec=164.0,
            text="발화 속도 설명",
            words=[{"word": "발화", "start": 154.0, "end": 154.2}],
        ),
        TimedUnit(
            unit_id=1,
            start_sec=164.0,
            end_sec=176.0,
            text="피드백을 제공합니다 현재까지 진행 현황입니다",
            words=[
                {"word": "피드백을", "start": 164.0, "end": 164.5},
                {"word": "제공합니다", "start": 171.0, "end": 171.5},
                {"word": "현재까지", "start": 174.0, "end": 174.4},
                {"word": "진행", "start": 174.5, "end": 174.8},
                {"word": "현황입니다", "start": 174.9, "end": 175.3},
            ],
        ),
        TimedUnit(
            unit_id=2,
            start_sec=176.5,
            end_sec=190.0,
            text="AI 엔진 진행 현황",
            words=[{"word": "AI", "start": 176.5, "end": 176.7}],
        ),
    ]
    slides = [
        [Keypoint(text="해결방법", importance=1.0, source="title")],
        [Keypoint(text="진행현황", importance=1.0, source="title")],
    ]

    refined = _refine_boundaries_with_title_cues(
        [0.0, 164.0, 190.0],
        units,
        [0, 1, 1],
        slides,
        config={"title_cue_refine_min_offset_sec": 1.0},
    )

    assert refined == [0.0, 174.5, 190.0]
