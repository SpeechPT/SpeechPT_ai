"""Whisper 전사 결과를 슬라이드별로 분할하는 모듈."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TranscriptSegment:
    slide_id: int
    start_sec: float
    end_sec: float
    text: str
    words: List[Dict]
    warning_flags: List[str] = field(default_factory=list)


def _normalize_boundaries(slide_change_times: List[float]) -> List[float]:
    if not slide_change_times:
        return []

    cleaned = sorted({float(t) for t in slide_change_times if t is not None})
    if len(cleaned) < 2:
        return cleaned
    return cleaned


def align_transcript(words: List[Dict], slide_change_times: List[float]) -> List[TranscriptSegment]:
    """슬라이드 경계 기준으로 word-level 타임스탬프를 안정적으로 분할한다.

    반환 세그먼트 수는 가능하면 슬라이드 구간 수와 맞춘다.
    단어가 없는 슬라이드도 빈 세그먼트로 유지해 이후 단계의 slide_id 정합성이 깨지지 않게 한다.
    """
    boundaries = _normalize_boundaries(slide_change_times)
    if len(boundaries) < 2:
        if not words:
            return []
        text = " ".join(str(w.get("word", "")).strip() for w in words).strip()
        start = float(words[0].get("start", 0.0))
        end = float(words[-1].get("end", start))
        return [TranscriptSegment(slide_id=1, start_sec=start, end_sec=end, text=text, words=words)]

    if not words:
        segments: List[TranscriptSegment] = []
        for idx in range(len(boundaries) - 1):
            segments.append(
                TranscriptSegment(
                    slide_id=idx + 1,
                    start_sec=boundaries[idx],
                    end_sec=boundaries[idx + 1],
                    text="",
                    words=[],
                    warning_flags=["empty_segment"],
                )
            )
        return segments

    buckets: List[List[Dict]] = [[] for _ in range(len(boundaries) - 1)]
    first_boundary = boundaries[0]
    last_boundary = boundaries[-1]

    for word in words:
        t = float(word.get("start", 0.0))
        if t < first_boundary:
            bucket_idx = 0
        elif t >= last_boundary:
            bucket_idx = len(buckets) - 1
        else:
            bucket_idx = 0
            for idx in range(len(boundaries) - 1):
                start = boundaries[idx]
                end = boundaries[idx + 1]
                if start <= t < end:
                    bucket_idx = idx
                    break
        buckets[bucket_idx].append(word)

    segments = []
    for idx, bucket in enumerate(buckets):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        warning_flags: List[str] = []
        if not bucket:
            warning_flags.append("empty_segment")
            text = ""
        else:
            text = " ".join(str(w.get("word", "")).strip() for w in bucket).strip()
            actual_start = float(bucket[0].get("start", start))
            actual_end = float(bucket[-1].get("end", actual_start))
            if actual_start > end:
                warning_flags.append("late_segment_start")
            if actual_end < start:
                warning_flags.append("early_segment_end")
        segments.append(
            TranscriptSegment(
                slide_id=idx + 1,
                start_sec=start,
                end_sec=end,
                text=text,
                words=bucket,
                warning_flags=warning_flags,
            )
        )
    return segments


def main():
    parser = argparse.ArgumentParser(description="Align transcript words to slide segments")
    parser.add_argument("words_json", help="Path to JSON list of words ({'word','start','end'})")
    parser.add_argument("slide_changes", help="Comma-separated slide change times (sec)")
    args = parser.parse_args()

    words = json.loads(Path(args.words_json).read_text())
    slide_times = [float(t) for t in args.slide_changes.split(",") if t]
    segments = align_transcript(words, slide_times)
    for seg in segments:
        logger.info("slide %d: %.2f-%.2f | %d words", seg.slide_id, seg.start_sec, seg.end_sec, len(seg.words))


if __name__ == "__main__":
    from pathlib import Path

    main()
