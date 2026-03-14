"""문서(PDF/PPT)에서 슬라이드별 텍스트와 시각 요소를 구조화하여 추출하는 모듈."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class VisualItem:
    """슬라이드 내 시각 요소 메타데이터."""

    item_id: str
    slide_id: int
    item_type: str
    source: str  # "pdf" | "ppt"
    bbox: List[float] | None = None
    raw_text: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlideContent:
    """슬라이드 단위 텍스트/시각 요소 컨테이너."""

    slide_id: int
    text: str
    title: str
    bullet_points: List[str]
    visual_captions: List[str] = field(default_factory=list)
    visual_items: List[VisualItem] = field(default_factory=list)


def _extract_bullets(lines: List[str]) -> List[str]:
    bullets = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "•", "·", "*", "●")):
            bullets.append(stripped.lstrip("-•·*● "))
    return bullets


def _summarize_visual_items(items: List[VisualItem]) -> List[str]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item.item_type] = counts.get(item.item_type, 0) + 1
    captions = []
    for item_type, count in sorted(counts.items()):
        captions.append(f"{item_type} x{count}")
    return captions


def _extract_pdf_visual_items(page: fitz.Page, slide_id: int) -> List[VisualItem]:
    items: List[VisualItem] = []

    for idx, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]
        rects = page.get_image_rects(xref)
        bbox = None
        if rects:
            r = rects[0]
            bbox = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
        items.append(
            VisualItem(
                item_id=f"s{slide_id}_img_{idx}",
                slide_id=slide_id,
                item_type="image",
                source="pdf",
                bbox=bbox,
                metadata={"xref": xref},
            )
        )

    drawings = page.get_drawings()
    if drawings:
        items.append(
            VisualItem(
                item_id=f"s{slide_id}_chart_candidate_1",
                slide_id=slide_id,
                item_type="chart_candidate",
                source="pdf",
                metadata={"drawings_count": len(drawings)},
            )
        )

    try:
        tables = page.find_tables()
        if getattr(tables, "tables", None):
            for idx, _ in enumerate(tables.tables, start=1):
                items.append(
                    VisualItem(
                        item_id=f"s{slide_id}_table_{idx}",
                        slide_id=slide_id,
                        item_type="table",
                        source="pdf",
                    )
                )
    except Exception:
        # PyMuPDF 버전에 따라 find_tables 미지원일 수 있어 무시한다.
        pass

    return items


def parse_pdf(path: Path) -> List[SlideContent]:
    doc = fitz.open(path)
    slides: List[SlideContent] = []
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            bullet_points = _extract_bullets(lines)
            visual_items = _extract_pdf_visual_items(page, i)
            slides.append(
                SlideContent(
                    slide_id=i,
                    text="\n".join(lines),
                    title=title,
                    bullet_points=bullet_points,
                    visual_captions=_summarize_visual_items(visual_items),
                    visual_items=visual_items,
                )
            )
    finally:
        doc.close()
    return slides


def _shape_to_item_type(shape) -> str | None:
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        return "image"
    if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
        return "table"
    if shape.shape_type == MSO_SHAPE_TYPE.CHART:
        return "chart"
    smart_art = getattr(MSO_SHAPE_TYPE, "SMART_ART", None)
    if smart_art is not None and shape.shape_type == smart_art:
        return "diagram"
    return None


def _extract_ppt_visual_items(slide, slide_id: int) -> List[VisualItem]:
    items: List[VisualItem] = []
    for idx, shape in enumerate(slide.shapes, start=1):
        item_type = _shape_to_item_type(shape)
        if item_type is None:
            continue

        bbox = [float(shape.left), float(shape.top), float(shape.left + shape.width), float(shape.top + shape.height)]
        metadata: Dict[str, Any] = {"shape_id": shape.shape_id}
        if item_type == "chart" and hasattr(shape, "chart"):
            metadata["chart_type"] = str(shape.chart.chart_type)
        items.append(
            VisualItem(
                item_id=f"s{slide_id}_{item_type}_{idx}",
                slide_id=slide_id,
                item_type=item_type,
                source="ppt",
                bbox=bbox,
                metadata=metadata,
            )
        )
    return items


def parse_ppt(path: Path) -> List[SlideContent]:
    prs = Presentation(path)
    slides: List[SlideContent] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts: List[str] = []
        title_text = ""
        bullet_points: List[str] = []
        visual_items = _extract_ppt_visual_items(slide, i)
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            if shape.is_placeholder and shape.placeholder_format.type in {1, 3}:
                title_text = shape.text.strip()
            for paragraph in shape.text_frame.paragraphs:
                p_text = "".join(run.text for run in paragraph.runs).strip()
                if not p_text:
                    continue
                texts.append(p_text)
                if paragraph.level > 0 or p_text.startswith(("-", "•", "·", "*", "●")):
                    bullet_points.append(p_text.lstrip("-•·*● "))
        slides.append(
            SlideContent(
                slide_id=i,
                text="\n".join(texts),
                title=title_text,
                bullet_points=bullet_points,
                visual_captions=_summarize_visual_items(visual_items),
                visual_items=visual_items,
            )
        )
    return slides


def parse_document(path: str | Path) -> List[SlideContent]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".ppt", ".pptx"}:
        return parse_ppt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def main():
    parser = argparse.ArgumentParser(description="Parse PDF/PPT into slide texts and visual item metadata")
    parser.add_argument("input", type=str, help="Path to PDF or PPTX file")
    args = parser.parse_args()

    slides = parse_document(args.input)
    for slide in slides:
        logger.info(
            "Slide %d | title=%s | bullets=%d | visuals=%d | visual_captions=%s",
            slide.slide_id,
            slide.title[:30],
            len(slide.bullet_points),
            len(slide.visual_items),
            slide.visual_captions,
        )


if __name__ == "__main__":
    main()
