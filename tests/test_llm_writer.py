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


class BadScopeClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, max_output_tokens: int):
        return {
            "overall_comment": "키포인트와 커버리지를 확인하세요.",
            "strengths": [],
            "priority_actions": [
                "슬라이드 5에서 핵심 키포인트 설명이 부족하니 더 자세히 말하세요.",
                "슬라이드 8에서 말이 멈춘 시간을 줄이세요.",
            ],
            "slide_comments": [
                {"slide_id": 5, "comment": "핵심 키포인트 설명이 부족합니다.", "evidence": "coverage 낮음", "action": "보완하세요."},
                {"slide_id": 8, "comment": "말이 멈춘 시간이 깁니다.", "evidence": "silence", "action": "연결 멘트를 넣으세요."},
            ],
            "detailed_report": {
                "top_issues": [{"rank": 1, "issue": "키포인트 부족", "affected_slides": [5], "action": "보완"}],
                "slide_by_slide_feedback": [
                    {"slide_id": 5, "observed": "내용 전달 부족", "recommendation": "보완"},
                    {"slide_id": 8, "observed": "말이 멈춘 시간이 김", "recommendation": "연결 멘트"},
                ],
            },
        }


def test_llm_prompt_requires_grounded_actionable_feedback():
    assert "입력으로 제공된 CE/AE 분석 JSON만 근거" in SYSTEM_PROMPT
    assert "새 축으로 확장하지 마세요" in SYSTEM_PROMPT
    assert "사용자가 실제로 따라 할 수 있는 행동" in SYSTEM_PROMPT
    assert "근거 없는 추상문" in USER_PROMPT
    assert "무엇을/어디서/어떻게" in USER_PROMPT
    assert "detailed_report" in USER_PROMPT
    assert "transcript_segments" in SYSTEM_PROMPT
    assert "highlight_sections에 없는 슬라이드" in SYSTEM_PROMPT
    assert "내용 부족/누락/문제라고 단정 금지" in USER_PROMPT
    assert 'issues가 ["pitch_shift"]뿐이면 음 높이 변화만' in SYSTEM_PROMPT
    assert "content_gap/title_missing/bullet_missing/visual_not_explained issue가 없는 슬라이드" in USER_PROMPT
    assert "목소리 톤이 크게 오르내림" in SYSTEM_PROMPT
    assert "음 높이/피치/키포인트/커버리지/alignment/confidence" in USER_PROMPT


def test_build_llm_feedback_passes_compact_report_to_prompt():
    client = DummyClient()
    report = {
        "overall_scores": {"content_coverage": 42.0},
        "global_summary": {"total_slides": 2},
        "highlight_sections": [{"slide_id": 2, "issues": ["silence_excess"]}],
        "per_slide_detail": [{"slide_id": 2, "silence_ratio": 0.35}],
        "transcript_segments": [
            {
                "slide_id": 2,
                "start_sec": 10.0,
                "end_sec": 20.0,
                "text": "이 구간은 실제 발표 대사입니다.",
                "words": [{"word": "불필요한", "start": 10.0}],
            }
        ],
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
    assert payload["transcript_segments"][0]["text"] == "이 구간은 실제 발표 대사입니다."
    assert "words" not in payload["transcript_segments"][0]


def test_build_llm_feedback_truncates_transcript_segments():
    client = DummyClient()
    long_text = "가" * 50
    report = {
        "transcript_segments": [{"slide_id": 1, "text": long_text}],
        "alignment": {},
    }

    build_llm_feedback(
        report,
        config={"enabled": True, "model": "dummy", "max_transcript_chars_per_slide": 10},
        client=client,
    )

    compact_json = client.user_prompt.split("분석 JSON:", 1)[1].strip()
    payload = json.loads(compact_json)
    assert payload["transcript_segments"][0]["text"] == "가" * 9 + "…"


def test_build_llm_feedback_sanitizes_issue_scope_and_internal_terms():
    report = {
        "overall_scores": {"delivery_stability": 88.0, "pacing_score": 92.0},
        "highlight_sections": [
            {"slide_id": 5, "issues": ["pitch_shift"]},
            {"slide_id": 8, "issues": ["silence_excess"]},
        ],
        "alignment": {},
    }

    feedback = build_llm_feedback(
        report,
        config={"enabled": True, "model": "dummy"},
        client=BadScopeClient(),
    )

    assert feedback is not None
    assert "키포인트" not in json.dumps(feedback, ensure_ascii=False)
    assert "커버리지" not in json.dumps(feedback, ensure_ascii=False)
    assert all("슬라이드 5" not in item for item in feedback["priority_actions"])
    assert [item["slide_id"] for item in feedback["slide_comments"]] == [8]
    assert feedback["strengths"]
