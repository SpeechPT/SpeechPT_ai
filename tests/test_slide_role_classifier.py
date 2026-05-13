from speechpt.coherence.document_parser import SlideContent
from speechpt.coherence.slide_role_classifier import classify_slide_roles
from speechpt.coherence.vlm_caption import VlmSlideCaption


def test_classify_slide_roles_uses_basic_heuristics():
    slides = [
        SlideContent(slide_id=1, text="프로젝트명", title="프로젝트명", bullet_points=[]),
        SlideContent(slide_id=2, text="개요", title="개요", bullet_points=[]),
        SlideContent(slide_id=3, text="핵심 결과 설명", title="실험 결과", bullet_points=["정확도 개선"]),
        SlideContent(slide_id=4, text="감사합니다", title="감사합니다", bullet_points=[]),
    ]

    roles = classify_slide_roles(slides)

    assert roles[1].role == "cover"
    assert roles[2].role == "section"
    assert roles[3].role == "content"
    assert roles[4].role == "thanks"
    assert roles[1].suppress_title_missing is True
    assert roles[3].coverage_weight == 1.0
    assert roles[4].coverage_weight == 0.0


def test_classify_slide_roles_prefers_vlm_slide_type():
    slides = [SlideContent(slide_id=1, text="표지처럼 짧음", title="짧음", bullet_points=[])]
    captions = [
        VlmSlideCaption(
            slide_id=1,
            slide_type="chart",
            role_in_flow="결과를 보여준다.",
            main_claim="수치가 개선되었다.",
            visual_kind="bar_chart",
            visual_summary="막대 그래프",
            entities=[],
            likely_keywords_in_speech=[],
        )
    ]

    roles = classify_slide_roles(slides, captions)

    assert roles[1].role == "content"
    assert roles[1].source == "vlm"
    assert roles[1].slide_type == "chart"

