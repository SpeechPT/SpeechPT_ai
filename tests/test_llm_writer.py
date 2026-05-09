import json

from speechpt.report.llm_writer import SYSTEM_PROMPT, USER_PROMPT, build_llm_feedback


class DummyClient:
    def __init__(self):
        self.system_prompt = ""
        self.user_prompt = ""

    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, max_output_tokens: int):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return {
            "overall_comment": "content coverage 42.0점을 근거로 핵심 설명 보완이 우선입니다.",
            "strengths": [],
            "priority_actions": ["슬라이드 2 시작 시 핵심 용어를 한 문장으로 먼저 말하세요."],
            "slide_comments": [{"slide_id": 2, "comment": "슬라이드 2는 침묵 비율 0.35가 높아 전환 멘트를 준비하세요."}],
        }


def test_llm_prompt_requires_grounded_actionable_feedback():
    assert "입력으로 제공된 CE/AE 분석 JSON만 근거" in SYSTEM_PROMPT
    assert "새 축으로 확장하지 마세요" in SYSTEM_PROMPT
    assert "사용자가 실제로 따라 할 수 있는 행동" in SYSTEM_PROMPT
    assert "근거 없는 추상문" in USER_PROMPT
    assert "무엇을/어디서/어떻게" in USER_PROMPT


def test_build_llm_feedback_passes_compact_report_to_prompt():
    client = DummyClient()
    report = {
        "overall_scores": {"content_coverage": 42.0},
        "global_summary": {"total_slides": 2},
        "highlight_sections": [{"slide_id": 2, "issues": ["silence_excess"]}],
        "per_slide_detail": [{"slide_id": 2, "silence_ratio": 0.35}],
        "alignment": {"mode": "auto", "strategy_used": "auto", "confidence": 0.04, "warnings": []},
    }

    feedback = build_llm_feedback(
        report,
        config={"enabled": True, "model": "dummy", "max_output_tokens": 500},
        client=client,
    )

    assert feedback is not None
    assert feedback["priority_actions"][0].startswith("슬라이드 2")
    assert "content_coverage" in client.user_prompt
    compact_json = client.user_prompt.split("분석 JSON:", 1)[1].strip()
    payload = json.loads(compact_json)
    assert payload["alignment"]["confidence"] == 0.04
