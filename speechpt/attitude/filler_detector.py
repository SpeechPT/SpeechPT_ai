"""간투사(filler word) 탐지 모듈.

STT 출력(단어 + 타임스탬프 리스트)을 받아서
발화 중 "어", "음", "그", "저", "뭐", "아" 같은 간투사를 탐지하고
슬라이드 섹션별로 집계한다.

Input (whisper_words 형식):
    [{"word": "안녕하세요", "start": 0.0, "end": 0.4}, ...]

Output:
    {
        "total_fillers": 12,
        "filler_rate": 0.08,          # 전체 단어 대비 간투사 비율
        "filler_words": [
            {"word": "어", "start": 3.2, "end": 3.4, "slide_id": 1},
            ...
        ],
        "per_slide": {
            1: {"count": 4, "rate": 0.10, "words": [...]},
            2: {"count": 2, "rate": 0.05, "words": [...]},
        }
    }
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

# 한국어 간투사 패턴 — 단독 출현 또는 짧은 반복형 포함
DEFAULT_FILLER_PATTERNS = [
    r"^어+$",       # 어, 어어, 어어어
    r"^음+$",       # 음, 음음
    r"^아+$",       # 아, 아아
    r"^그+$",       # 그
    r"^저+$",       # 저
    r"^뭐+$",       # 뭐
    r"^에+$",       # 에, 에에
    r"^이+$",       # 이
    r"^그래서$",
    r"^그러니까$",
    r"^사실$",
    r"^뭐랄까$",
    r"^있잖아$",
    r"^있잖아요$",
]


def _compile_patterns(patterns: Sequence[str] | None = None) -> List[Pattern[str]]:
    raw_patterns = list(DEFAULT_FILLER_PATTERNS)
    if patterns:
        raw_patterns.extend(str(pattern) for pattern in patterns)
    # Preserve order while removing duplicate pattern strings.
    deduped = list(dict.fromkeys(raw_patterns))
    return [re.compile(pattern) for pattern in deduped]


def is_filler(word: str, patterns: Sequence[str] | None = None) -> bool:
    w = word.strip().lower()
    return any(pattern.fullmatch(w) for pattern in _compile_patterns(patterns))


def detect_fillers(
    words: Sequence[Dict],
    slide_timestamps: Optional[List[float]] = None,
    patterns: Sequence[str] | None = None,
) -> Dict:
    """간투사를 탐지하고 슬라이드 섹션별로 집계한다.

    Args:
        words: [{"word": str, "start": float, "end": float}, ...]
        slide_timestamps: 슬라이드 경계 초 리스트. 예: [0, 40, 50, 90]
                          None이면 슬라이드 분류 없이 전체 집계만 반환.
        patterns: 추가 filler 정규식. 기본 한국어 filler 패턴에 합산된다.

    Returns:
        탐지 결과 dict.
    """
    # 슬라이드 구간 구성
    segments: List[Tuple[float, float, int]] = []  # (start, end, slide_id)
    if slide_timestamps and len(slide_timestamps) >= 2:
        ts = sorted(slide_timestamps)
        for i in range(len(ts) - 1):
            segments.append((ts[i], ts[i + 1], i + 1))

    def _slide_id_for(start: float) -> Optional[int]:
        for seg_start, seg_end, sid in segments:
            if seg_start <= start < seg_end:
                return sid
        return None

    compiled = _compile_patterns(patterns)
    filler_words = []
    for w in words:
        token = str(w.get("word", "")).strip().lower()
        if any(pattern.fullmatch(token) for pattern in compiled):
            entry = {
                "word": w["word"],
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
            }
            if segments:
                sid = _slide_id_for(entry["start"])
                if sid is not None:
                    entry["slide_id"] = sid
            filler_words.append(entry)

    total_words = len(words)
    total_fillers = len(filler_words)
    filler_rate = total_fillers / total_words if total_words > 0 else 0.0

    result: Dict = {
        "total_words": total_words,
        "total_fillers": total_fillers,
        "filler_rate": round(filler_rate, 4),
        "filler_words": filler_words,
    }

    if segments:
        per_slide: Dict[int, Dict] = {}
        for _, _, sid in segments:
            per_slide[sid] = {"count": 0, "words": []}

        slide_word_counts: Dict[int, int] = {sid: 0 for _, _, sid in segments}
        for w in words:
            sid = _slide_id_for(w.get("start", 0.0))
            if sid is not None:
                slide_word_counts[sid] += 1

        for fw in filler_words:
            sid = fw.get("slide_id")
            if sid is not None and sid in per_slide:
                per_slide[sid]["count"] += 1
                per_slide[sid]["words"].append(fw)

        for sid in per_slide:
            total_in_slide = slide_word_counts.get(sid, 0)
            count = per_slide[sid]["count"]
            per_slide[sid]["rate"] = round(count / total_in_slide, 4) if total_in_slide > 0 else 0.0

        result["per_slide"] = per_slide

    return result
