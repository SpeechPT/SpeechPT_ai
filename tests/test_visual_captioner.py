from speechpt.coherence.document_parser import VisualItem
from speechpt.coherence.visual_captioner import build_visual_captions


def test_build_visual_captions_prefers_ocr_text():
    items = [
        VisualItem(item_id="1", slide_id=1, item_type="chart", source="pdf", raw_text="매출 20% 증가", confidence=0.9),
        VisualItem(item_id="2", slide_id=1, item_type="image", source="pdf", raw_text="", confidence=0.0),
    ]
    captions = build_visual_captions(items, min_confidence=0.3)
    assert any("chart:" in c for c in captions)
    assert any("매출" in c for c in captions)


def test_build_visual_captions_fallback_to_counts():
    items = [
        VisualItem(item_id="1", slide_id=1, item_type="image", source="pdf", raw_text="", confidence=0.0),
        VisualItem(item_id="2", slide_id=1, item_type="image", source="pdf", raw_text="", confidence=0.0),
    ]
    captions = build_visual_captions(items, min_confidence=0.3)
    assert captions == ["image x2"]
