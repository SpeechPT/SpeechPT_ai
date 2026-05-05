import tempfile
from pathlib import Path

import fitz

from speechpt.coherence.document_parser import SlideContent
from speechpt.coherence.vlm_caption import (
    PROMPT_VERSION,
    caption_document_slides,
    render_pdf_slides_png,
    validate_caption_payload,
    validate_presentation_payload,
)


class FakeVisionClient:
    def __init__(self):
        self.calls = 0

    def caption_presentation(self, *, image_pngs, prompt: str, model: str, detail: str, max_tokens: int):
        self.calls += 1
        assert image_pngs
        assert "발표 슬라이드 분석가" in prompt
        assert detail == "low"
        return {
            "presentation": {
                "topic": "알파 제어 결과",
                "core_terminology": ["알파", "오차", "결과"],
                "sections": [{"name": "전체", "slide_indices": [1]}],
            },
            "slides": [
                {
                    "index": 1,
                    "slide_type": "chart",
                    "role_in_flow": "실험 결과를 설명한다.",
                    "main_claim": "알파 제어가 안정적으로 작동한다.",
                    "visual_kind": "line_chart",
                    "visual_summary": "알파 증가에 따라 오차가 감소하는 그래프",
                    "entities": ["alpha"],
                    "likely_keywords_in_speech": ["알파", "오차", "결과"],
                }
            ],
        }

    def caption_slide(self, *, image_png: bytes, prompt: str, model: str, detail: str):
        self.calls += 1
        assert image_png
        assert "발표 슬라이드 분석가" in prompt
        assert detail == "low"
        return {
            "presentation": {
                "topic": "알파 제어 결과",
                "core_terminology": ["알파"],
                "sections": [{"name": "전체", "slide_indices": [1]}],
            },
            "slides": [
                {
                    "index": 1,
                    "slide_type": "chart",
                    "role_in_flow": "실험 결과를 설명한다.",
                    "main_claim": "알파 제어가 안정적으로 작동한다.",
                    "visual_kind": "line_chart",
                    "visual_summary": "알파 증가에 따라 오차가 감소하는 그래프",
                    "entities": ["alpha"],
                    "likely_keywords_in_speech": ["알파", "오차", "결과"],
                }
            ],
        }


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Result Slide\nAlpha error decreases")
    doc.save(path)
    doc.close()


def test_validate_caption_payload_sanitizes_schema():
    caption = validate_caption_payload(
        {
            "slide_type": "unknown-type",
            "visual_caption": "  그래프 요약입니다.  ",
            "key_claims": ["주장", "주장", ""],
            "likely_keywords_in_speech": ["알파", "결과"],
        },
        slide_id=3,
        model="mock-model",
    )

    assert caption.slide_id == 3
    assert caption.slide_type == "content"
    assert caption.visual_caption == "그래프 요약입니다."
    assert caption.key_claims == ["주장"]
    assert caption.likely_keywords_in_speech == ["알파", "결과"]
    assert caption.prompt_version == PROMPT_VERSION


def test_render_pdf_slides_png_matches_page_count(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "one")
    doc.new_page().insert_text((72, 72), "two")
    doc.save(pdf_path)
    doc.close()

    images = render_pdf_slides_png(pdf_path, dpi=144)

    assert len(images) == 2
    assert all(image.startswith(b"\x89PNG") for image in images)


def test_validate_presentation_payload_rejects_bad_sections():
    payload = {
        "presentation": {
            "topic": "주제",
            "core_terminology": ["용어"],
            "sections": [{"name": "일부", "slide_indices": [1]}],
        },
        "slides": [
            {
                "index": 1,
                "slide_type": "content",
                "role_in_flow": "역할",
                "main_claim": "주장",
                "visual_kind": "none",
                "visual_summary": "",
                "entities": [],
                "likely_keywords_in_speech": [],
            },
            {
                "index": 2,
                "slide_type": "content",
                "role_in_flow": "역할",
                "main_claim": "주장",
                "visual_kind": "none",
                "visual_summary": "",
                "entities": [],
                "likely_keywords_in_speech": [],
            },
        ],
    }

    try:
        validate_presentation_payload(payload, slide_count=2, model="mock-model")
    except ValueError as exc:
        assert "sections" in str(exc)
    else:
        raise AssertionError("expected invalid sections to fail")


def test_caption_document_slides_uses_cache_hit_and_miss(tmp_path: Path):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf_path = Path(tmp.name)
    _make_pdf(pdf_path)
    slides = [SlideContent(slide_id=1, text="Result", title="Result", bullet_points=[])]
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "model": "mock-model",
        "detail": "low",
        "cache_enabled": True,
    }

    miss_client = FakeVisionClient()
    first = caption_document_slides(pdf_path, slides, cfg, client=miss_client)
    assert miss_client.calls == 1
    assert first[0].slide_type == "chart"

    hit_client = FakeVisionClient()
    second = caption_document_slides(pdf_path, slides, cfg, client=hit_client)
    assert hit_client.calls == 0
    assert second[0].visual_caption == first[0].visual_caption

    pdf_path.unlink()
