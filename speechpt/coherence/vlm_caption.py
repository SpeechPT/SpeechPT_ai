"""VLM slide captioning for CE auto-alignment only.

The generated captions are alignment signals. They must not be used as
presentation feedback or as additional coherence coverage targets.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

import fitz

from speechpt.coherence.document_parser import SlideContent
from speechpt.openai_utils import resolve_openai_api_key

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_DPI = 144

ALLOWED_SLIDE_TYPES = {
    "title",
    "section_header",
    "content",
    "chart",
    "table",
    "comparison",
    "conclusion",
    "thanks",
}

ALLOWED_VISUAL_KINDS = {
    "none",
    "bar_chart",
    "line_chart",
    "table",
    "diagram",
    "screenshot",
    "image",
    "flowchart",
}

VLM_PRESENTATION_PROMPT = """당신은 발표 슬라이드 분석가입니다. 주어진 발표의 모든 슬라이드 이미지를 순서대로 보고, 지정된 JSON 형식으로만 응답하세요. 각 슬라이드를 단독으로 보지 말고, 발표 전체의 흐름 안에서 어떤 역할을 하는지를 고려하세요. 슬라이드 순서를 임의로 바꾸지 마세요. JSON 외의 텍스트, 코드블록, 설명을 포함하지 마세요.

출력 JSON 형식:
{
  "presentation": {
    "topic": "발표 주제 (1문장, 한국어)",
    "core_terminology": ["발표 전체에서 일관되게 쓰이는 핵심 용어 5~10개"],
    "sections": [
      {"name": "섹션 이름", "slide_indices": [1, 2, 3]}
    ]
  },
  "slides": [
    {
      "index": 1,
      "slide_type": "title | section_header | content | chart | table | comparison | conclusion | thanks",
      "role_in_flow": "이 슬라이드가 발표 흐름에서 하는 역할 (1문장, 한국어)",
      "main_claim": "이 슬라이드의 핵심 주장 (1문장, 한국어)",
      "visual_kind": "none | bar_chart | line_chart | table | diagram | screenshot | image | flowchart",
      "visual_summary": "시각 자료가 보여주는 내용 (없으면 빈 문자열). 추세/비교/구조를 구체적으로.",
      "entities": ["고유명사/모델명/지표/숫자"],
      "likely_keywords_in_speech": ["이 슬라이드 위에서 발화에 등장할 가능성이 높은 단어/구 2~4개"]
    }
  ]
}

규칙:
- 이 정보는 CE 자동정렬 신호 전용입니다.
- 발표자를 평가하거나, 발표자가 무엇을 누락했는지 판단하지 마세요.
- 슬라이드에 보이지 않는 사실을 만들지 마세요.
- main_claim은 발표자가 말할 법한 자연스러운 한국어 설명문으로 작성하세요.
- likely_keywords_in_speech 필드명은 반드시 그대로 사용하세요.
"""

VLM_SINGLE_SLIDE_FALLBACK_PROMPT = VLM_PRESENTATION_PROMPT + "\n입력 이미지는 슬라이드 1장입니다. slides에는 index 1인 항목 하나만 넣으세요."

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class VlmSection:
    name: str
    slide_indices: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VlmPresentationCaption:
    topic: str
    core_terminology: List[str]
    sections: List[VlmSection]
    prompt_version: str = PROMPT_VERSION
    model: str = DEFAULT_MODEL

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [section.to_dict() for section in self.sections]
        return payload


@dataclass
class VlmSlideCaption:
    """Validated VLM output for one slide."""

    slide_id: int
    slide_type: str
    role_in_flow: str
    main_claim: str
    visual_kind: str
    visual_summary: str
    entities: List[str]
    likely_keywords_in_speech: List[str]
    prompt_version: str = PROMPT_VERSION
    model: str = DEFAULT_MODEL
    source: str = "vlm"

    @property
    def visual_caption(self) -> str:
        return self.visual_summary

    @property
    def key_claims(self) -> List[str]:
        return [self.main_claim] if self.main_claim else []

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["index"] = self.slide_id
        return payload


@dataclass
class VlmCaptionResult:
    presentation: VlmPresentationCaption
    slides: List[VlmSlideCaption]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "presentation": self.presentation.to_dict(),
            "slides": [slide.to_dict() for slide in self.slides],
        }


class VisionCaptionClient(Protocol):
    def caption_presentation(self, *, image_pngs: Sequence[bytes], prompt: str, model: str, detail: str, max_tokens: int) -> Dict[str, Any]:
        """Return raw VLM JSON payload for all slide images."""

    def caption_slide(self, *, image_png: bytes, prompt: str, model: str, detail: str) -> Dict[str, Any]:
        """Return raw VLM JSON payload for one slide image."""


class OpenAIResponsesVisionClient:
    """Minimal stdlib client for OpenAI Responses vision calls."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/responses",
        timeout_sec: float = 60.0,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required for VLM captioning.")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_sec = timeout_sec

    def caption_presentation(self, *, image_pngs: Sequence[bytes], prompt: str, model: str, detail: str, max_tokens: int) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for idx, image_png in enumerate(image_pngs, start=1):
            data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
            content.append({"type": "input_text", "text": f"Slide {idx}"})
            content.append({"type": "input_image", "image_url": data_url, "detail": detail})
        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        return _parse_json_object(self._request(payload))

    def caption_slide(self, *, image_png: bytes, prompt: str, model: str, detail: str) -> Dict[str, Any]:
        return self.caption_presentation(
            image_pngs=[image_png],
            prompt=prompt,
            model=model,
            detail=detail,
            max_tokens=700,
        )

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
            raise RuntimeError(f"OpenAI vision request failed: {exc.code} {detail_text}") from exc
        return _extract_response_text(response_payload)


def _extract_response_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    parts: List[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = _JSON_FENCE_RE.sub("", text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("VLM response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("VLM response JSON must be an object.")
    return payload


def _safe_str(value: Any, max_len: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text[:max_len].strip()


def _safe_str_list(value: Any, max_items: int, max_len: int = 120) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        text = _safe_str(item, max_len=max_len)
        key = text.lower()
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _pdf_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    root = os.environ.get("SPEECHPT_CACHE_DIR")
    if root:
        return Path(root) / "vlm_captions"
    return Path("cache") / "vlm_captions"


def _cache_path(cache_dir: Path, document_path: Path, *, model: str, dpi: int) -> Path:
    key = {
        "pdf_sha256": _pdf_sha256(document_path),
        "dpi": dpi,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def render_pdf_slides_png(document_path: str | Path, *, dpi: int = DEFAULT_DPI) -> List[bytes]:
    path = Path(document_path)
    zoom = float(dpi) / 72.0
    doc = fitz.open(path)
    try:
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images.append(pix.tobytes("png"))
        return images
    finally:
        doc.close()


def render_pdf_slide_png(document_path: str | Path, slide_id: int, config: Dict | None = None) -> bytes:
    cfg = config or {}
    return render_pdf_slides_png(document_path, dpi=int(cfg.get("dpi", cfg.get("render_dpi", DEFAULT_DPI))))[slide_id - 1]


def validate_caption_payload(payload: Dict[str, Any], *, slide_id: int, model: str) -> VlmSlideCaption:
    """Backward-compatible validator for a single slide payload."""

    slide_type = _safe_str(payload.get("slide_type"), max_len=32).lower()
    if slide_type not in ALLOWED_SLIDE_TYPES:
        slide_type = "content"
    visual_kind = _safe_str(payload.get("visual_kind"), max_len=32).lower()
    if visual_kind not in ALLOWED_VISUAL_KINDS:
        visual_kind = "none"
    main_claim = _safe_str(payload.get("main_claim") or (payload.get("key_claims") or [""])[0])
    visual_summary = _safe_str(payload.get("visual_summary") or payload.get("visual_caption"))
    return VlmSlideCaption(
        slide_id=slide_id,
        slide_type=slide_type,
        role_in_flow=_safe_str(payload.get("role_in_flow")),
        main_claim=main_claim,
        visual_kind=visual_kind,
        visual_summary=visual_summary,
        entities=_safe_str_list(payload.get("entities"), max_items=10, max_len=80),
        likely_keywords_in_speech=_safe_str_list(payload.get("likely_keywords_in_speech"), max_items=8, max_len=80),
        prompt_version=PROMPT_VERSION,
        model=model,
    )


def validate_presentation_payload(payload: Dict[str, Any], *, slide_count: int, model: str) -> VlmCaptionResult:
    presentation_payload = payload.get("presentation")
    slides_payload = payload.get("slides")
    if not isinstance(presentation_payload, dict) or not isinstance(slides_payload, list):
        raise ValueError("VLM payload must include presentation object and slides list.")
    if len(slides_payload) != slide_count:
        raise ValueError(f"VLM slides length mismatch: expected {slide_count}, got {len(slides_payload)}.")

    slides: List[VlmSlideCaption] = []
    seen_indices = []
    for expected_idx, item in enumerate(slides_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each VLM slide payload must be an object.")
        index = int(item.get("index", item.get("slide_id", 0)))
        if index != expected_idx:
            raise ValueError(f"VLM slide index mismatch: expected {expected_idx}, got {index}.")
        seen_indices.append(index)
        slides.append(validate_caption_payload(item, slide_id=index, model=model))

    sections = _validate_sections(presentation_payload.get("sections"), slide_count=slide_count)
    presentation = VlmPresentationCaption(
        topic=_safe_str(presentation_payload.get("topic")),
        core_terminology=_safe_str_list(presentation_payload.get("core_terminology"), max_items=12, max_len=80),
        sections=sections,
        prompt_version=PROMPT_VERSION,
        model=model,
    )
    if sorted(seen_indices) != list(range(1, slide_count + 1)):
        raise ValueError("VLM slide indices do not cover 1..N.")
    return VlmCaptionResult(presentation=presentation, slides=slides)


def _validate_sections(value: Any, *, slide_count: int) -> List[VlmSection]:
    if not isinstance(value, list):
        raise ValueError("presentation.sections must be a list.")
    sections: List[VlmSection] = []
    covered: List[int] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each section must be an object.")
        indices = [int(idx) for idx in item.get("slide_indices", [])]
        if not indices:
            continue
        if any(idx < 1 or idx > slide_count for idx in indices):
            raise ValueError("Section slide_indices out of range.")
        covered.extend(indices)
        sections.append(VlmSection(name=_safe_str(item.get("name"), max_len=80), slide_indices=indices))
    if sorted(covered) != list(range(1, slide_count + 1)):
        raise ValueError("presentation.sections must cover slide indices 1..N exactly.")
    return sections


def _read_cache(path: Path, *, slide_count: int, model: str) -> VlmCaptionResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("presentation", {}).get("prompt_version") != PROMPT_VERSION:
            return None
        return validate_presentation_payload(payload, slide_count=slide_count, model=model)
    except Exception:
        logger.warning("Ignoring invalid VLM caption cache: %s", path)
        return None


def _write_cache(path: Path, result: VlmCaptionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(result.to_dict(), tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _build_client(config: Dict) -> VisionCaptionClient | None:
    api_key = resolve_openai_api_key(config)
    if not api_key:
        logger.warning("VLM captioning enabled but OPENAI_API_KEY is not configured; falling back without captions.")
        return None
    return OpenAIResponsesVisionClient(
        api_key=api_key,
        base_url=str(config.get("base_url", "https://api.openai.com/v1/responses")),
        timeout_sec=float(config.get("timeout_sec", 60.0)),
    )


def _fallback_caption_slide(
    *,
    client: VisionCaptionClient,
    image_png: bytes,
    slide_id: int,
    model: str,
    detail: str,
) -> VlmSlideCaption | None:
    try:
        payload = client.caption_slide(image_png=image_png, prompt=VLM_SINGLE_SLIDE_FALLBACK_PROMPT, model=model, detail=detail)
        slide_payload = payload.get("slides", [payload])[0] if isinstance(payload.get("slides"), list) else payload
        if isinstance(slide_payload, dict):
            slide_payload = dict(slide_payload)
            slide_payload["index"] = slide_id
        return validate_caption_payload(slide_payload, slide_id=slide_id, model=model)
    except Exception as exc:
        logger.warning("VLM fallback caption failed for slide %s: %s", slide_id, exc)
        return None


def caption_document(
    document_path: str | Path,
    slides: Sequence[SlideContent],
    config: Dict | None = None,
    *,
    client: VisionCaptionClient | None = None,
) -> VlmCaptionResult | None:
    cfg = config or {}
    path = Path(document_path)
    if path.suffix.lower() != ".pdf":
        logger.info("VLM captioning currently supports PDF input only; skipping %s.", path.suffix)
        return None

    model = str(cfg.get("model", DEFAULT_MODEL))
    detail = str(cfg.get("detail", "low"))
    dpi = int(cfg.get("dpi", cfg.get("render_dpi", DEFAULT_DPI)))
    cache_enabled = bool(cfg.get("cache_enabled", True))
    cache_dir = Path(cfg.get("cache_dir", _default_cache_dir()))
    cache_file = _cache_path(cache_dir, path, model=model, dpi=dpi)
    slide_count = len(slides)
    if slide_count <= 0:
        return None

    if cache_enabled:
        cached = _read_cache(cache_file, slide_count=slide_count, model=model)
        if cached is not None:
            return cached

    active_client = client or _build_client(cfg)
    if active_client is None:
        return None

    images = render_pdf_slides_png(path, dpi=dpi)
    if len(images) != slide_count:
        raise ValueError(f"Rendered slide count mismatch: expected {slide_count}, got {len(images)}.")

    max_tokens = int(cfg.get("max_tokens", slide_count * 400 + 800))
    try:
        payload = active_client.caption_presentation(
            image_pngs=images,
            prompt=str(cfg.get("prompt", VLM_PRESENTATION_PROMPT)),
            model=model,
            detail=detail,
            max_tokens=max_tokens,
        )
        result = validate_presentation_payload(payload, slide_count=slide_count, model=model)
    except Exception as exc:
        logger.warning("VLM batch caption failed; retrying per-slide fallback: %s", exc)
        fallback_slides = []
        for idx, image in enumerate(images, start=1):
            caption = _fallback_caption_slide(client=active_client, image_png=image, slide_id=idx, model=model, detail=detail)
            if caption is None:
                return None
            fallback_slides.append(caption)
        result = VlmCaptionResult(
            presentation=VlmPresentationCaption(
                topic="",
                core_terminology=[],
                sections=[VlmSection(name="전체", slide_indices=list(range(1, slide_count + 1)))],
                model=model,
            ),
            slides=fallback_slides,
        )

    if cache_enabled:
        _write_cache(cache_file, result)
    return result


def caption_document_slides(
    document_path: str | Path,
    slides: Sequence[SlideContent],
    config: Dict | None = None,
    *,
    client: VisionCaptionClient | None = None,
) -> List[VlmSlideCaption]:
    """Caption PDF slides with a VLM, using per-document cache and safe fallback."""

    result = caption_document(document_path, slides, config, client=client)
    return result.slides if result is not None else []


__all__ = [
    "PROMPT_VERSION",
    "VLM_PRESENTATION_PROMPT",
    "VlmCaptionResult",
    "VlmPresentationCaption",
    "VlmSection",
    "VlmSlideCaption",
    "caption_document",
    "caption_document_slides",
    "render_pdf_slide_png",
    "render_pdf_slides_png",
    "validate_caption_payload",
    "validate_presentation_payload",
]
