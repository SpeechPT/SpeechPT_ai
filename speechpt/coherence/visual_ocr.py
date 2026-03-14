"""Visual item OCR extraction utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import fitz

from .document_parser import SlideContent, VisualItem


def _crop_page_region(page: fitz.Page, bbox: List[float]) -> fitz.Pixmap | None:
    if not bbox or len(bbox) != 4:
        return None
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    if rect.is_empty or rect.is_infinite:
        return None
    return page.get_pixmap(clip=rect)


def _ocr_with_easyocr(image_bytes: bytes, languages: Sequence[str]) -> List[Dict]:
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("easyocr is required for visual OCR. Install with `pip install easyocr`.") from exc

    reader = easyocr.Reader(list(languages), gpu=False)
    results = reader.readtext(image_bytes)
    out = []
    for _, text, conf in results:
        token = str(text).strip()
        if token:
            out.append({"text": token, "confidence": float(conf)})
    return out


def enrich_slides_with_visual_ocr(slides: Sequence[SlideContent], document_path: str | Path, config: Dict | None = None) -> None:
    """Fill VisualItem.raw_text/confidence using OCR for PDF visual items.

    The function mutates `slides` in-place.
    """
    cfg = config or {}
    engine = str(cfg.get("ocr_engine", "easyocr"))
    languages = cfg.get("ocr_languages", ["ko", "en"])

    path = Path(document_path)
    if path.suffix.lower() != ".pdf":
        # PPT 렌더링 기반 OCR은 2차 단계에서 추가한다.
        return

    doc = fitz.open(path)
    try:
        for slide in slides:
            page_idx = slide.slide_id - 1
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            for item in slide.visual_items:
                if item.source != "pdf" or not item.bbox:
                    continue
                pix = _crop_page_region(page, item.bbox)
                if pix is None:
                    continue
                image_bytes = pix.tobytes("png")
                if engine != "easyocr":
                    raise ValueError(f"Unsupported ocr_engine: {engine}")
                ocr_items = _ocr_with_easyocr(image_bytes, languages)
                if not ocr_items:
                    continue
                texts = [x["text"] for x in ocr_items]
                confs = [x["confidence"] for x in ocr_items]
                item.raw_text = " ".join(texts).strip()
                item.confidence = float(sum(confs) / len(confs))
    finally:
        doc.close()
