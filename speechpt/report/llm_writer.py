"""Optional LLM rewriting layer for final report output."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Protocol

from speechpt.openai_utils import resolve_openai_api_key
from speechpt.report.score_mapping import map_coverage_score

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1/responses"
CONTENT_ISSUES = {"content_gap", "title_missing", "bullet_missing", "visual_not_explained"}
INTERNAL_TERM_REPLACEMENTS = {
    "content coverage": "내용 연결",
    "Content coverage": "내용 연결",
    "키포인트": "핵심 내용",
    "커버리지": "내용 연결",
    "음 높이": "목소리 톤",
    "피치": "목소리 톤",
    "dwell": "설명 시간",
    "Dwell": "설명 시간",
    "alignment confidence": "분석 신뢰도",
    "alignment": "분석",
    "confidence": "분석 신뢰도",
    "semantic": "의미 연결",
    "threshold": "기준값",
    "probe": "분석 모델",
}
CONTENT_PROBLEM_PATTERN = re.compile(r"(내용.*(부족|누락)|핵심.*(부족|누락)|키포인트|커버리지)")
SLIDE_ID_PATTERN = re.compile(r"슬라이드\s*(\d+)")

SYSTEM_PROMPT = """당신은 SpeechPT의 발표 코칭 리포트 작성자입니다.
목표는 사용자가 다음 녹음에서 바로 고칠 수 있는 구체적인 발표 피드백 리포트 JSON을 작성하는 것입니다.

근거 규칙:
- 입력으로 제공된 CE/AE 분석 JSON만 근거로 작성하세요.
- CE는 슬라이드 핵심 내용과 발화의 정합성 신호입니다. 분석 JSON의 content_connection_score는 사용자 노출용으로 변환된 점수이며, 원본 raw 점수는 제공되지 않습니다. 슬라이드 내용을 모두 설명했는지 채점하는 새 축으로 확장하지 마세요.
- AE는 발화 속도, 침묵, dwell, 간투사, pitch/energy 변화 신호입니다. 인성/능력/성실성 같은 추정은 금지합니다.
- VLM/visual/likely_keywords 계열 신호가 있더라도 사용자에게 "말했어야 할 정답 키워드"처럼 단정하지 마세요.
- alignment.warnings 또는 confidence가 낮으면 정렬이 불확실하다는 점을 반영해 단정적인 표현을 줄이세요.
- transcript_segments가 있으면 실제 발화 문장을 근거로 사용하되, 사용자의 의도나 지식을 추정하지 마세요.
- content_connection_score나 content_connection_label이 낮아도 "발표를 못했다"처럼 단정하지 말고, STT/정렬/표현 차이 가능성을 함께 고려하세요.
- priority_actions, slide_comments, detailed_report.top_issues는 highlight_sections에 실제로 등장한 issue를 최우선 근거로 삼으세요.
- highlight_sections에 없는 슬라이드에 대해 "부족", "누락", "문제" 같은 부정 피드백을 새로 만들지 마세요. per_slide_detail은 issue의 수치 근거를 보강할 때만 사용하세요.
- 특정 슬라이드의 content_connection_label이 상대적으로 좋고 highlight issue가 없으면, 그 슬라이드를 내용 부족 사례로 지목하지 마세요.
- 각 슬라이드의 피드백 주제는 highlight_sections[].issues에 있는 issue로 제한하세요. 예를 들어 issues가 ["pitch_shift"]뿐이면 음 높이 변화만 말하고, content_gap/title_missing/bullet_missing/visual_not_explained가 없으면 내용 부족이나 누락을 언급하지 마세요.
- content_gap, title_missing, bullet_missing, visual_not_explained가 없는 슬라이드에는 "핵심 내용 누락", "키포인트 부족", "내용 전달 부족" 같은 표현을 쓰지 마세요.

사용자 표현 규칙:
- 내부 용어를 그대로 쓰지 마세요. pitch_shift는 "목소리 톤이 크게 오르내림", silence_ratio는 "말이 멈춘 시간", dwell은 "슬라이드에 사용한 시간", coverage는 "슬라이드 내용과 설명의 연결", alignment confidence는 "분석 신뢰도"로 풀어 쓰세요.
- "음 높이", "피치", "키포인트", "커버리지", "alignment", "confidence", "semantic", "threshold", "dwell", "probe" 같은 단어를 사용자 문장에 노출하지 마세요.
- 추천 문장은 문제명보다 행동을 먼저 말하세요. 예: "슬라이드 8 시작 전에 '마지막으로 기대 효과를 말씀드리겠습니다'처럼 짧은 연결 문장을 넣어보세요."
- 사용자가 듣기에 어색한 분석어 대신 발표자가 바로 따라 할 수 있는 말하기 행동으로 표현하세요.

작성 규칙:
- 모든 문장은 한국어 존댓말로 작성하세요.
- 추상적인 표현만 쓰지 말고, 가능한 경우 입력 JSON의 수치나 슬라이드 번호를 근거로 포함하세요.
- "좋습니다", "보완하세요"처럼 단독으로 끝나는 일반론은 금지합니다.
- 개선 액션은 사용자가 실제로 따라 할 수 있는 행동으로 쓰세요. 예: "슬라이드 3 시작 시 핵심 용어를 한 문장으로 먼저 말하세요."
- 전체 평가는 5~8문장으로 작성하고, 점수 해석/신뢰도/가장 큰 개선 우선순위를 포함하세요.
- 강점은 실제 입력 근거가 있는 경우에만 작성하세요. 근거가 약하면 "상대적으로 안정적인 부분"처럼 제한적으로 표현하세요.
- slide_comments는 문제가 큰 슬라이드 위주로 작성하되, 각 항목에 evidence와 action을 함께 넣으세요.
- detailed_report에는 점수별 해석, 주요 문제, 슬라이드별 상세 피드백, 다음 연습 계획을 충분히 작성하세요.
- 데이터가 애매하면 "가장 큰 문제"를 억지로 만들지 말고, 확인이 필요한 항목으로 표현하세요.
- 반드시 JSON 객체만 출력하세요. 코드블록, 설명, 마크다운을 출력하지 마세요."""

USER_PROMPT = """아래 SpeechPT 분석 결과를 사용자에게 보여줄 최종 코멘트로 요약하세요.

출력 품질 기준:
- overall_comment: 전체 점수, alignment warning/confidence, 가장 큰 개선 우선순위를 5~8문장으로 설명
- strengths: 실제 근거가 있는 강점 1~2개. 없으면 억지로 만들지 말고 빈 배열 허용
- priority_actions: 우선순위 높은 개선 행동 5개. 각 항목은 "무엇을/어디서/어떻게"가 드러나야 함
- slide_comments: highlight_sections, per_slide_detail, transcript_segments에서 문제가 큰 슬라이드를 선택. 각 항목은 comment/evidence/action 포함
- detailed_report.score_interpretation: content_coverage, delivery_stability, pacing_score 각각을 사용자가 이해할 수 있게 해석
- 분석 JSON의 overall_scores.content_connection_score는 사용자에게 보여줄 변환 점수입니다. 원본 raw coverage처럼 퍼센트로 다시 계산하거나 슬라이드별 raw 점수를 추정하지 마세요.
- per_slide_detail에는 슬라이드별 content_connection_label만 제공합니다. 이 라벨을 임의의 퍼센트 숫자로 바꾸지 마세요.
- detailed_report.top_issues: 가장 중요한 문제 3~5개를 근거와 함께 정리
- detailed_report.slide_by_slide_feedback: 가능한 한 주요 슬라이드마다 관찰/근거/수정 행동/연습 문장을 작성
- detailed_report.practice_plan: 다음 1회 녹음에서 할 일, 3회 반복 연습 계획, 체크리스트 작성
- priority_actions/top_issues/slide_comments는 highlight_sections에 있는 issue를 우선 사용
- highlight_sections에 없는 슬라이드는 중립적 관찰만 가능하며, 내용 부족/누락/문제라고 단정 금지
- 슬라이드별 comment는 해당 highlight_sections[].issues 범위를 넘지 말 것. pitch_shift만 있는 슬라이드에는 pitch만, silence_excess만 있는 슬라이드에는 침묵만 설명
- content_gap/title_missing/bullet_missing/visual_not_explained issue가 없는 슬라이드는 내용 부족 사례로 언급 금지
- 사용자에게 내부 분석어를 노출하지 말 것: 음 높이/피치/키포인트/커버리지/alignment/confidence/semantic/threshold/dwell/probe 금지
- technical metric은 쉬운 말로 변환: 목소리 톤, 말이 멈춘 시간, 설명 시간, 슬라이드와 말의 연결, 분석 신뢰도

표현 금지:
- "전반적으로 좋습니다", "더 자세히 설명하세요", "명확히 전달하세요"처럼 근거 없는 추상문
- 입력 JSON에 없는 새 점수, 새 평가축, 슬라이드 충실도/누락 주제 채점
- 사용자에게 VLM caption, likely_keywords_in_speech, 내부 alignment 신호명을 그대로 노출

출력 JSON 형식:
{{
  "overall_comment": "전체 발표에 대한 5~8문장 상세 요약. 수치, 신뢰도, 주요 근거를 포함.",
  "strengths": ["근거가 있는 잘한 점 1", "근거가 있는 잘한 점 2"],
  "priority_actions": ["구체적 개선 액션 1", "구체적 개선 액션 2", "구체적 개선 액션 3", "구체적 개선 액션 4", "구체적 개선 액션 5"],
  "slide_comments": [
    {{
      "slide_id": 1,
      "comment": "해당 슬라이드의 핵심 피드백.",
      "evidence": "근거 수치나 실제 발화 요약.",
      "action": "다음 녹음에서 바로 할 행동."
    }}
  ],
  "detailed_report": {{
    "confidence_note": "정렬 신뢰도와 해석 주의점.",
    "score_interpretation": {{
      "content_coverage": "내용 전달 점수 해석.",
      "delivery_stability": "전달 안정성 점수 해석.",
      "pacing_score": "발표 속도 점수 해석."
    }},
    "top_issues": [
      {{
        "rank": 1,
        "issue": "핵심 문제명",
        "evidence": "근거",
        "affected_slides": [1, 2],
        "action": "수정 행동"
      }}
    ],
    "slide_by_slide_feedback": [
      {{
        "slide_id": 1,
        "observed": "관찰된 현상",
        "evidence": "점수/발화/시간 근거",
        "recommendation": "수정 권장",
        "practice_script": "연습용 예시 문장"
      }}
    ],
    "practice_plan": {{
      "next_recording": "다음 녹음 1회에서 우선 적용할 계획",
      "three_repetition_plan": "3회 반복 연습 계획",
      "checklist": ["녹음 전 확인 1", "녹음 전 확인 2", "녹음 전 확인 3"]
    }}
  }}
}}

분석 JSON:
{report_json}
"""


class TextGenerationClient(Protocol):
    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, max_output_tokens: int) -> Dict[str, Any]:
        """Return a parsed JSON object generated by the model."""


class OpenAIResponsesTextClient:
    """Minimal stdlib OpenAI Responses client for text-only JSON output."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout_sec: float = 60.0):
        if not api_key:
            raise ValueError("OpenAI API key is required for LLM report generation.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_sec = timeout_sec

    def generate_json(self, *, system_prompt: str, user_prompt: str, model: str, max_output_tokens: int) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        return _parse_json_object(self._request(payload))

    def _request(self, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI report request failed: {exc.code} {detail_text}") from exc
        return _extract_response_text(response_payload)


def _extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("LLM report response JSON must be an object.")
    return payload


def _truncate_text(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _compact_transcript_segments(report: Dict[str, Any], *, max_slides: int, max_chars: int) -> list[Dict[str, Any]]:
    compact_segments: list[Dict[str, Any]] = []
    for segment in report.get("transcript_segments", [])[:max_slides]:
        if not isinstance(segment, dict):
            continue
        compact_segments.append(
            {
                "slide_id": segment.get("slide_id"),
                "start_sec": segment.get("start_sec"),
                "end_sec": segment.get("end_sec"),
                "text": _truncate_text(segment.get("text", ""), max_chars),
                "warning_flags": segment.get("warning_flags", []),
            }
        )
    return compact_segments


def _round_number(value: Any, ndigits: int = 2) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), ndigits)


def _content_connection_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 85.0:
        return "높음"
    if score >= 70.0:
        return "양호"
    if score >= 50.0:
        return "보통"
    return "확인 필요"


def _mapped_content_score(scores: Dict[str, Any], raw_key: str, user_key: str) -> float | None:
    user_score = _round_number(scores.get(user_key))
    if user_score is not None:
        return user_score
    raw_score = scores.get(raw_key)
    if isinstance(raw_score, (int, float)):
        return round(map_coverage_score(float(raw_score)), 2)
    return None


def _compact_overall_scores(scores: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    content_score = _mapped_content_score(scores, "content_coverage", "content_coverage_user")
    content_score_all = _mapped_content_score(scores, "content_coverage_all", "content_coverage_user_all")
    if content_score is not None:
        compact["content_connection_score"] = content_score
        compact["content_connection_label"] = _content_connection_label(content_score)
    if content_score_all is not None:
        compact["content_connection_score_all"] = content_score_all
    for key in ("content_scored_slide_count", "delivery_stability", "pacing_score"):
        if key in scores:
            compact[key] = scores[key]
    return compact


def _compact_global_summary(summary: Dict[str, Any], compact_scores: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "total_slides",
        "content_scored_slide_count",
        "mean_speech_rate",
        "mean_silence_ratio",
        "total_change_points",
        "delivery_stability_components",
        "summary_text",
    ):
        if key in summary:
            compact[key] = summary[key]
    if "content_connection_score" in compact_scores:
        compact["avg_content_connection_score"] = compact_scores["content_connection_score"]
        compact["avg_content_connection_label"] = compact_scores.get("content_connection_label")
    return compact


def _compact_slide_detail(slide: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "slide_id",
        "slide_role",
        "coverage_weight",
        "speech_rate",
        "silence_ratio",
        "filler_count",
        "word_count",
        "words_per_sec",
        "dwell_sec",
        "dwell_ratio",
        "delivery_stability",
        "trend",
        "anomalies",
        "ae_probe",
    ):
        if key in slide:
            compact[key] = slide[key]
    raw_coverage = slide.get("coverage")
    if isinstance(raw_coverage, (int, float)):
        raw_pct = float(raw_coverage) * 100.0 if float(raw_coverage) <= 1.0 else float(raw_coverage)
        user_score = map_coverage_score(raw_pct)
        compact["content_connection_label"] = _content_connection_label(user_score)
    return compact


def _compact_slide_details(report: Dict[str, Any], max_slides: int) -> list[Dict[str, Any]]:
    compact: list[Dict[str, Any]] = []
    for slide in report.get("per_slide_detail", [])[:max_slides]:
        if isinstance(slide, dict):
            compact.append(_compact_slide_detail(slide))
    return compact


def _compact_report(report: Dict[str, Any], max_slides: int = 12, max_transcript_chars: int = 900) -> Dict[str, Any]:
    compact_scores = _compact_overall_scores(report.get("overall_scores", {}))
    return {
        "overall_scores": compact_scores,
        "global_summary": _compact_global_summary(report.get("global_summary", {}), compact_scores),
        "highlight_sections": report.get("highlight_sections", [])[:max_slides],
        "per_slide_detail": _compact_slide_details(report, max_slides),
        "transcript_segments": _compact_transcript_segments(
            report,
            max_slides=max_slides,
            max_chars=max_transcript_chars,
        ),
        "alignment": {
            key: report.get("alignment", {}).get(key)
            for key in ("mode", "strategy_used", "confidence", "warnings")
            if key in report.get("alignment", {})
        },
    }


def _allowed_issues_by_slide(compact_report: Dict[str, Any]) -> Dict[int, set[str]]:
    allowed: Dict[int, set[str]] = {}
    for section in compact_report.get("highlight_sections", []):
        if not isinstance(section, dict):
            continue
        try:
            slide_id = int(section.get("slide_id"))
        except (TypeError, ValueError):
            continue
        allowed[slide_id] = {str(issue) for issue in section.get("issues", [])}
    return allowed


def _replace_internal_terms(value: Any) -> Any:
    if isinstance(value, str):
        text = value
        for source, target in INTERNAL_TERM_REPLACEMENTS.items():
            text = text.replace(source, target)
        return text
    if isinstance(value, list):
        return [_replace_internal_terms(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_internal_terms(item) for key, item in value.items()}
    return value


def _has_content_issue(slide_id: int | None, allowed: Dict[int, set[str]]) -> bool:
    if slide_id is None:
        return True
    return bool(allowed.get(slide_id, set()) & CONTENT_ISSUES)


def _extract_slide_id(item: Any) -> int | None:
    if isinstance(item, dict):
        try:
            return int(item.get("slide_id"))
        except (TypeError, ValueError):
            pass
        affected = item.get("affected_slides")
        if isinstance(affected, list) and len(affected) == 1:
            try:
                return int(affected[0])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(item, str):
        match = SLIDE_ID_PATTERN.search(item)
        if match:
            return int(match.group(1))
    return None


def _contains_content_problem(item: Any) -> bool:
    if isinstance(item, str):
        return bool(CONTENT_PROBLEM_PATTERN.search(item))
    if isinstance(item, dict):
        return any(_contains_content_problem(value) for value in item.values())
    if isinstance(item, list):
        return any(_contains_content_problem(value) for value in item)
    return False


def _filter_issue_scoped_items(items: Any, allowed: Dict[int, set[str]]) -> Any:
    if not isinstance(items, list):
        return items
    filtered = []
    for item in items:
        slide_id = _extract_slide_id(item)
        if slide_id is not None and not _has_content_issue(slide_id, allowed) and _contains_content_problem(item):
            continue
        filtered.append(item)
    return filtered


def _ensure_strengths(feedback: Dict[str, Any], compact_report: Dict[str, Any]) -> None:
    strengths = feedback.get("strengths")
    if isinstance(strengths, list) and strengths:
        return
    scores = compact_report.get("overall_scores", {})
    generated: list[str] = []
    delivery = scores.get("delivery_stability")
    pacing = scores.get("pacing_score")
    if isinstance(delivery, (int, float)) and delivery >= 80:
        generated.append("말하기 흐름은 전반적으로 안정적으로 측정되었습니다.")
    if isinstance(pacing, (int, float)) and pacing >= 80:
        generated.append("발표 속도는 청중이 따라가기 무리 없는 범위로 측정되었습니다.")
    feedback["strengths"] = generated[:2]


def _sanitize_llm_feedback(feedback: Dict[str, Any], compact_report: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = _replace_internal_terms(feedback)
    if not isinstance(sanitized, dict):
        return feedback

    allowed = _allowed_issues_by_slide(compact_report)
    sanitized["priority_actions"] = _filter_issue_scoped_items(sanitized.get("priority_actions", []), allowed)
    sanitized["slide_comments"] = _filter_issue_scoped_items(sanitized.get("slide_comments", []), allowed)

    detailed = sanitized.get("detailed_report")
    if isinstance(detailed, dict):
        detailed["top_issues"] = _filter_issue_scoped_items(detailed.get("top_issues", []), allowed)
        detailed["slide_by_slide_feedback"] = _filter_issue_scoped_items(
            detailed.get("slide_by_slide_feedback", []),
            allowed,
        )
    _ensure_strengths(sanitized, compact_report)
    return sanitized


def build_llm_feedback(
    report: Dict[str, Any],
    config: Dict[str, Any] | None = None,
    client: TextGenerationClient | None = None,
) -> Dict[str, Any] | None:
    """Generate optional final LLM feedback. Returns None on disabled/fallback."""
    cfg = config or {}
    if not cfg.get("enabled", False):
        return None
    resolved_client = client
    if resolved_client is None:
        api_key = resolve_openai_api_key(cfg)
        if not api_key:
            logger.warning("LLM report enabled but OpenAI API key is not configured; skipping LLM output.")
            return None
        resolved_client = OpenAIResponsesTextClient(
            api_key=api_key,
            base_url=str(cfg.get("base_url", DEFAULT_BASE_URL)),
            timeout_sec=float(cfg.get("timeout_sec", 60.0)),
        )

    compact = _compact_report(
        report,
        max_slides=int(cfg.get("max_slides", 12)),
        max_transcript_chars=int(cfg.get("max_transcript_chars_per_slide", 900)),
    )
    user_prompt = USER_PROMPT.format(report_json=json.dumps(compact, ensure_ascii=False, indent=2))
    try:
        feedback = resolved_client.generate_json(
            system_prompt=str(cfg.get("system_prompt", SYSTEM_PROMPT)),
            user_prompt=user_prompt,
            model=str(cfg.get("model", DEFAULT_MODEL)),
            max_output_tokens=int(cfg.get("max_output_tokens", 4000)),
        )
        return _sanitize_llm_feedback(feedback, compact)
    except Exception:
        logger.exception("LLM report generation failed; returning deterministic report only.")
        return None
