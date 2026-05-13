"""Classify slide roles for report aggregation and conservative feedback gating."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Sequence

from speechpt.coherence.document_parser import SlideContent
from speechpt.coherence.vlm_caption import VlmSlideCaption


CONTENT_ROLE = "content"
NON_CONTENT_ROLES = {"cover", "section", "thanks"}
THANKS_RE = re.compile(r"(감사|thank|q\s*&?\s*a|질문)", re.IGNORECASE)


@dataclass(frozen=True)
class SlideRole:
    slide_id: int
    role: str
    source: str
    reason: str
    slide_type: str | None = None
    coverage_weight: float = 1.0
    suppress_title_missing: bool = False
    suppress_content_issues: bool = False

    @property
    def is_content(self) -> bool:
        return self.role == CONTENT_ROLE

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["is_content"] = self.is_content
        return payload


def _non_content_role(slide_id: int, role: str, source: str, reason: str, slide_type: str | None = None) -> SlideRole:
    return SlideRole(
        slide_id=slide_id,
        role=role,
        source=source,
        reason=reason,
        slide_type=slide_type,
        coverage_weight=0.0,
        suppress_title_missing=True,
        suppress_content_issues=True,
    )


def _content_role(slide_id: int, source: str, reason: str, slide_type: str | None = None) -> SlideRole:
    return SlideRole(
        slide_id=slide_id,
        role=CONTENT_ROLE,
        source=source,
        reason=reason,
        slide_type=slide_type,
        coverage_weight=1.0,
        suppress_title_missing=False,
        suppress_content_issues=False,
    )


def _role_from_vlm(slide: SlideContent, caption: VlmSlideCaption, *, is_first: bool) -> SlideRole:
    slide_type = caption.slide_type
    if slide_type == "thanks":
        return _non_content_role(slide.slide_id, "thanks", "vlm", "vlm_slide_type=thanks", slide_type)
    if slide_type == "title":
        role = "cover" if is_first else "section"
        return _non_content_role(slide.slide_id, role, "vlm", "vlm_slide_type=title", slide_type)
    if slide_type == "section_header":
        return _non_content_role(slide.slide_id, "section", "vlm", "vlm_slide_type=section_header", slide_type)
    return _content_role(slide.slide_id, "vlm", f"vlm_slide_type={slide_type}", slide_type)


def _is_short_title_only(slide: SlideContent) -> bool:
    title = slide.title.strip()
    text = " ".join(slide.text.split())
    combined = f"{title} {text}".strip()
    if not combined:
        return False
    if slide.bullet_points:
        return False
    return len(combined) <= 45 and len(title) <= 35


def _role_from_heuristic(slide: SlideContent, *, is_first: bool, is_last: bool) -> SlideRole:
    title = slide.title.strip()
    text = " ".join([title, slide.text]).strip()
    if is_first:
        return _non_content_role(slide.slide_id, "cover", "heuristic", "first_slide")
    if is_last and THANKS_RE.search(text):
        return _non_content_role(slide.slide_id, "thanks", "heuristic", "last_slide_thanks_cue")
    if _is_short_title_only(slide):
        return _non_content_role(slide.slide_id, "section", "heuristic", "short_title_only")
    return _content_role(slide.slide_id, "heuristic", "default_content")


def classify_slide_roles(
    slides: Sequence[SlideContent],
    vlm_captions: Sequence[VlmSlideCaption] | None = None,
) -> Dict[int, SlideRole]:
    caption_map = {caption.slide_id: caption for caption in (vlm_captions or [])}
    last_slide_id = slides[-1].slide_id if slides else None
    roles: Dict[int, SlideRole] = {}
    for idx, slide in enumerate(slides):
        caption = caption_map.get(slide.slide_id)
        is_first = idx == 0
        is_last = slide.slide_id == last_slide_id
        if caption is not None:
            roles[slide.slide_id] = _role_from_vlm(slide, caption, is_first=is_first)
        else:
            roles[slide.slide_id] = _role_from_heuristic(slide, is_first=is_first, is_last=is_last)
    return roles
