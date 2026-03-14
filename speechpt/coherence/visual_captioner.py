"""Visual item to caption conversion rules."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .document_parser import VisualItem


def _short(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def build_visual_captions(items: Sequence[VisualItem], min_confidence: float = 0.3, max_text_len: int = 80) -> List[str]:
    captions: List[str] = []
    counts: Dict[str, int] = {}

    for item in items:
        counts[item.item_type] = counts.get(item.item_type, 0) + 1
        if item.raw_text and item.confidence >= min_confidence:
            captions.append(f"{item.item_type}: {_short(item.raw_text, max_text_len)}")
            continue
        if item.item_type == "chart" and "chart_type" in item.metadata:
            captions.append(f"chart: {item.metadata['chart_type']}")

    if not captions:
        for item_type, count in sorted(counts.items()):
            captions.append(f"{item_type} x{count}")

    # stable unique order
    seen = set()
    deduped: List[str] = []
    for cap in captions:
        k = cap.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(cap)
    return deduped
