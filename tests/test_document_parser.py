import tempfile
from pathlib import Path

import fitz
from pptx import Presentation

from speechpt.coherence.document_parser import parse_pdf, parse_ppt


def test_parse_pdf_extracts_title_and_text():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Title\n- point1\n- point2")
        doc.save(tmp.name)
        doc.close()
        slides = parse_pdf(Path(tmp.name))
    assert slides[0].title == "Title"
    assert "point1" in slides[0].text
    assert isinstance(slides[0].visual_items, list)
    assert isinstance(slides[0].visual_captions, list)
    Path(tmp.name).unlink()


def test_parse_ppt_reads_shapes():
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "My Title"
    slide.placeholders[1].text = "- bullet"
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as tmp:
        prs.save(tmp.name)
        slides = parse_ppt(Path(tmp.name))
    assert slides[0].title == "My Title"
    assert "bullet" in slides[0].bullet_points[0]
    assert isinstance(slides[0].visual_items, list)
    assert isinstance(slides[0].visual_captions, list)
    Path(tmp.name).unlink()
