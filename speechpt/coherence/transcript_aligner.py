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


def align_transcript(words: List[Dict], slide_change_times: List[float]) -> List[TranscriptSegment]:
    """슬라이드 전환 시점 기준으로 word-level 타임스탬프를 분할한다."""
    if not words:
        return []
    sorted_times = sorted(slide_change_times)
    boundaries = sorted_times + [float("inf")]
    segments: List[TranscriptSegment] = []
    current_words: List[Dict] = []
    slide_idx = 0
    slide_id = 1

    def flush_segment():
        nonlocal current_words, slide_id
        if not current_words:
            return
        text = " ".join(w["word"] for w in current_words).strip()
        start = current_words[0].get("start", 0.0)
        end = current_words[-1].get("end", start)
        segments.append(TranscriptSegment(slide_id=slide_id, start_sec=start, end_sec=end, text=text, words=current_words))
        current_words = []

    for w in words:
        t = w.get("start", 0.0)
        while t >= boundaries[slide_idx]:
            flush_segment()
            slide_idx += 1
            slide_id += 1
        current_words.append(w)

    flush_segment()
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
