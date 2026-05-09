"""슬라이드 경계 자동/하이브리드 정렬 모듈."""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import kss
import numpy as np
from kiwipiepy import Kiwi
from sklearn.metrics.pairwise import cosine_similarity

from speechpt.coherence.coherence_scorer import _get_model
from speechpt.coherence.keypoint_extractor import Keypoint

logger = logging.getLogger(__name__)

_ALIGNMENT_NORMALIZE_RE = re.compile(r"\s+")
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|[가-힣]{2,}|[0-9]+(?:\.[0-9]+)?%?")
_KIWI: Kiwi | None = None

DEFAULT_ALIGNMENT_CONFIG: Dict[str, Any] = {
    "pause_threshold_sec": 0.9,
    "chunk_min_duration_sec": 2.0,
    "chunk_max_duration_sec": 8.0,
    "chunk_min_words": 3,
    "chunk_max_words": 36,
    "min_first_chunk_sec": 2.0,
    "min_last_chunk_sec": 2.0,
    "switch_penalty": 0.08,
    "skip_penalty": 0.15,
    "extra_skip_penalty": 0.4,
    "start_skip_penalty": 2.0,
    "allow_skip": True,
    "max_slide_jump": 2,
    "min_dwell_units": 2,
    "stay_bonus": 0.05,
    "progress_penalty": 0.35,
    "lag_penalty_multiplier": 1.6,
    "anchor_cover_slide": True,
    "cover_min_window": 8.0,
    "cover_max_window": 20.0,
    "cover_ratio": 0.15,
    # Backward-compatible aliases for older config files.
    "cover_min_duration_sec": 8.0,
    "cover_window_max_sec": 20.0,
    "cover_window_ratio": 0.15,
    "anchor_thanks_slide": True,
    "thanks_search_start_ratio": 0.85,
    "closing_cues": ["감사합니다", "들어주셔", "이상입니", "마치겠습니", "감사드립니"],
    "short_title_max_words": 2,
    "short_title_max_chars": 6,
    "short_title_weight": 0.55,
    "default_title_weight": 0.35,
    "semantic_weight": 0.7,
    "lexical_weight": 0.2,
    "title_hit_weight": 0.1,
    "body_semantic_weight": 0.5,
    "keyword_semantic_weight": 0.3,
    "visual_semantic_weight": 0.2,
    "use_multi_signal_similarity": True,
    "confidence_min_similarity": 0.35,
    "confidence_min_margin": 0.1,
    "smoothing_enabled": True,
    "smoothing_margin_threshold": 0.1,
    "smoothing_max_units": 2,
    "smoothing_preserve_assigned_slides": True,
    "absorb_pause_chunks_enabled": True,
    "absorb_min_chunk_sec": 3.0,
    "absorb_edge_max_sec": 9.0,
    "absorb_max_chunk_utts": 2,
    "absorb_margin_threshold": 0.05,
    "cover_prefer_first_unit": True,
    "cover_first_unit_min_sec": 3.0,
    "cover_end_overrun_sec": 2.0,
    "cover_stop_on_next_title": True,
    "robust_fallback_enabled": True,
    "robust_fallback_confidence_threshold": 0.025,
    "robust_fallback_on_duplicate": True,
    "robust_fallback_on_unassigned": True,
    "robust_fallback_min_units_per_slide": 1,
    "robust_progress_penalty": 0.7,
    "robust_stay_bonus": 0.02,
    "robust_segmental_dp_enabled": True,
    "robust_min_segment_sec": 12.0,
    "robust_min_segment_ratio": 0.25,
    "robust_duration_penalty_weight": 0.25,
    "robust_short_segment_penalty_weight": 1.2,
    "robust_segment_progress_penalty": 0.35,
    "robust_boundary_pause_bonus": 0.08,
    "low_confidence_warning_threshold": 0.03,
    "title_cue_boundary_refinement_enabled": True,
    "title_cue_refine_min_offset_sec": 1.0,
    "title_cue_search_max_units": 4,
    "title_cue_allow_first_token_fallback": True,
    "vlm_semantic_mix_weight": 0.35,
    "vlm_title_claim_weight": 0.30,
    "vlm_title_role_weight": 0.20,
    "vlm_default_claim_weight": 0.50,
    "vlm_default_role_weight": 0.30,
    "vlm_default_visual_weight": 0.20,
    "vlm_visual_claim_weight": 0.40,
    "vlm_visual_role_weight": 0.20,
    "vlm_visual_visual_weight": 0.40,
    "vlm_boundary_influence_enabled": False,
}


@dataclass
class SlideRepresentation:
    title_emb: np.ndarray | None
    body_emb: np.ndarray | None
    keyword_emb: np.ndarray | None
    visual_emb: np.ndarray | None
    claim_emb: np.ndarray | None
    role_emb: np.ndarray | None
    vlm_visual_emb: np.ndarray | None
    title_weight: float
    title_tokens: set[str]
    all_keywords: set[str]
    slide_type: str | None = None
    visual_kind: str | None = None
    has_vlm: bool = False
    is_visual_heavy: bool = False


@dataclass
class EdgeAnchors:
    cover_end_sec: float
    thanks_start_sec: float
    cover_anchored: bool = False
    thanks_anchored: bool = False


def _alignment_config(config: Dict | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_ALIGNMENT_CONFIG)
    cfg.update(config or {})
    if config:
        if "cover_min_duration_sec" in config and "cover_min_window" not in config:
            cfg["cover_min_window"] = config["cover_min_duration_sec"]
        if "cover_window_max_sec" in config and "cover_max_window" not in config:
            cfg["cover_max_window"] = config["cover_window_max_sec"]
        if "cover_window_ratio" in config and "cover_ratio" not in config:
            cfg["cover_ratio"] = config["cover_window_ratio"]
    return cfg


def _has_vlm_keypoints(slide_keypoints: Sequence[Sequence[Keypoint]]) -> bool:
    return any(kp.source.startswith("vlm_") for keypoints in slide_keypoints for kp in keypoints)


def _without_vlm_keypoints(slide_keypoints: Sequence[Sequence[Keypoint]]) -> List[List[Keypoint]]:
    return [[kp for kp in keypoints if not kp.source.startswith("vlm_")] for keypoints in slide_keypoints]


def _config_float(config: Dict[str, Any], key: str, *aliases: str) -> float:
    for candidate in (key, *aliases):
        if candidate in config:
            return float(config[candidate])
    return float(DEFAULT_ALIGNMENT_CONFIG[key])


@dataclass
class TimedUnit:
    unit_id: int
    start_sec: float
    end_sec: float
    text: str
    words: List[Dict] = field(default_factory=list)


@dataclass
class AlignmentResult:
    mode: str
    strategy_used: str
    final_boundaries: List[float]
    provided_boundaries: List[float] = field(default_factory=list)
    proposed_boundaries: List[float] = field(default_factory=list)
    confidence: float | None = None
    low_confidence_segments: List[Dict] = field(default_factory=list)
    unit_assignments: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _clean_boundaries(boundaries: Sequence[float] | None) -> List[float]:
    if not boundaries:
        return []
    cleaned = []
    for boundary in boundaries:
        if boundary is None:
            continue
        cleaned.append(float(boundary))
    return sorted(set(cleaned))


def _word_start(word: Dict) -> float:
    return float(word.get("start", 0.0))


def _word_end(word: Dict) -> float:
    return float(word.get("end", _word_start(word)))


def _word_text(word: Dict) -> str:
    return str(word.get("word", "")).strip()


def _transcript_bounds(words: Sequence[Dict]) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0
    return _word_start(words[0]), _word_end(words[-1])


def _normalize_alignment_text(text: str) -> str:
    return _ALIGNMENT_NORMALIZE_RE.sub("", text).strip()


def _get_kiwi() -> Kiwi | None:
    global _KIWI
    if _KIWI is not None:
        return _KIWI
    try:
        _KIWI = Kiwi()
        return _KIWI
    except Exception:  # pragma: no cover - defensive fallback
        logger.warning("Kiwi tokenizer unavailable; falling back to regex tokenization.")
        return None


def tokenize_nouns(text: str) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    kiwi = _get_kiwi()
    if kiwi is not None:
        try:
            nouns = [token.form.lower() for token in kiwi.tokenize(normalized) if token.tag.startswith("N") or token.tag.startswith("SL") or token.tag.startswith("SN")]
            if nouns:
                return nouns
        except Exception:  # pragma: no cover - defensive fallback
            logger.warning("Kiwi tokenization failed; falling back to regex tokenization.")
    return [token.lower() for token in _FALLBACK_TOKEN_RE.findall(normalized)]


def is_short_title(title: str, config: Dict | None = None) -> bool:
    cfg = _alignment_config(config)
    compact = _normalize_alignment_text(title)
    if not compact:
        return False
    word_count = len(str(title).split())
    return word_count <= int(cfg["short_title_max_words"]) or len(compact) <= int(cfg["short_title_max_chars"])


def normalize_manual_boundaries(
    provided_boundaries: Sequence[float] | None,
    slide_count: int,
    words: Sequence[Dict],
) -> tuple[List[float], List[str]]:
    cleaned = _clean_boundaries(provided_boundaries)
    warnings: List[str] = []
    if slide_count <= 0:
        return [], warnings
    if not cleaned:
        raise ValueError("Manual alignment requires slide_timestamps.")

    start_hint, end_hint = _transcript_bounds(words)

    if len(cleaned) == slide_count + 1:
        if cleaned[0] > start_hint + 0.5:
            warnings.append("manual_start_after_transcript_start")
        if cleaned[-1] < end_hint - 0.5:
            warnings.append("manual_end_before_transcript_end")
        return cleaned, warnings

    if len(cleaned) == slide_count:
        if cleaned[0] > start_hint + 0.5:
            warnings.append("prepended_transcript_start")
            return [start_hint] + cleaned, warnings
        warnings.append("appended_transcript_end")
        return cleaned + [max(cleaned[-1], end_hint)], warnings

    if len(cleaned) > slide_count + 1:
        warnings.append("trimmed_extra_boundaries")
        trimmed = cleaned[: slide_count + 1]
        return trimmed, warnings

    raise ValueError(
        f"Expected {slide_count} or {slide_count + 1} slide boundaries for {slide_count} slides, got {len(cleaned)}."
    )


def _finalize_unit(raw_units: List[TimedUnit], words: List[Dict]) -> None:
    if not words:
        return
    raw_units.append(
        TimedUnit(
            unit_id=len(raw_units),
            start_sec=_word_start(words[0]),
            end_sec=_word_end(words[-1]),
            text=" ".join(_word_text(word) for word in words).strip(),
            words=list(words),
        )
    )


def _split_words_by_sentences(words: Sequence[Dict]) -> List[List[Dict]]:
    if not words:
        return []

    transcript = " ".join(_word_text(word) for word in words).strip()
    if not transcript:
        return [list(words)]

    try:
        sentences = [sentence.strip() for sentence in kss.split_sentences(transcript) if sentence.strip()]
    except Exception:  # pragma: no cover - defensive fallback
        logger.exception("KSS sentence split failed during auto alignment; falling back to a single chunk.")
        return [list(words)]
    if not sentences:
        return [list(words)]

    buckets: List[List[Dict]] = []
    word_idx = 0
    for sentence in sentences:
        target = _normalize_alignment_text(sentence)
        if not target:
            continue
        bucket: List[Dict] = []
        consumed = ""
        while word_idx < len(words):
            word = words[word_idx]
            bucket.append(word)
            consumed += _normalize_alignment_text(_word_text(word))
            word_idx += 1
            if consumed == target:
                break
            if len(consumed) >= len(target):
                break
        if bucket:
            buckets.append(bucket)

    if word_idx < len(words):
        if buckets:
            buckets[-1].extend(words[word_idx:])
        else:
            buckets.append(list(words[word_idx:]))

    return [bucket for bucket in buckets if bucket]


def _split_sentence_bucket(words: Sequence[Dict], config: Dict | None = None) -> List[List[Dict]]:
    cfg = _alignment_config(config)
    if not words:
        return []

    pause_threshold = float(cfg.get("pause_threshold_sec", 0.9))
    max_duration = float(cfg.get("chunk_max_duration_sec", 8.0))
    max_words = int(cfg.get("chunk_max_words", 36))

    split_buckets: List[List[Dict]] = []
    current_words: List[Dict] = []
    for word in words:
        if not current_words:
            current_words.append(word)
            continue
        prev_word = current_words[-1]
        gap = _word_start(word) - _word_end(prev_word)
        duration = _word_end(prev_word) - _word_start(current_words[0])
        if gap >= pause_threshold or duration >= max_duration or len(current_words) >= max_words:
            split_buckets.append(current_words)
            current_words = [word]
            continue
        current_words.append(word)

    if current_words:
        split_buckets.append(current_words)
    return split_buckets


def build_timed_units(words: Sequence[Dict], config: Dict | None = None) -> List[TimedUnit]:
    cfg = _alignment_config(config)
    if not words:
        return []

    max_duration = float(cfg.get("chunk_max_duration_sec", 8.0))
    min_duration = float(cfg.get("chunk_min_duration_sec", 2.0))
    min_words = int(cfg.get("chunk_min_words", 3))

    ordered = sorted(words, key=_word_start)
    raw_units: List[TimedUnit] = []
    sentence_buckets = _split_words_by_sentences(ordered)
    for sentence_words in sentence_buckets:
        for split_words in _split_sentence_bucket(sentence_words, cfg):
            _finalize_unit(raw_units, list(split_words))

    if not raw_units:
        return []

    merged: List[TimedUnit] = []
    for unit in raw_units:
        duration = unit.end_sec - unit.start_sec
        if (
            merged
            and (duration < min_duration or len(unit.words) < min_words)
            and (merged[-1].end_sec - merged[-1].start_sec) < max_duration * 1.5
        ):
            prev = merged[-1]
            prev.words.extend(unit.words)
            prev.end_sec = unit.end_sec
            prev.text = f"{prev.text} {unit.text}".strip()
            continue
        merged.append(unit)

    merged = _force_merge_edge_units(merged, cfg)

    for idx, unit in enumerate(merged):
        unit.unit_id = idx
    return merged


def _merge_unit_pair(first: TimedUnit, second: TimedUnit, unit_id: int) -> TimedUnit:
    words = list(first.words) + list(second.words)
    return TimedUnit(
        unit_id=unit_id,
        start_sec=first.start_sec,
        end_sec=second.end_sec,
        text=f"{first.text} {second.text}".strip(),
        words=words,
    )


def _force_merge_edge_units(units: Sequence[TimedUnit], config: Dict | None = None) -> List[TimedUnit]:
    cfg = _alignment_config(config)
    merged = list(units)
    if len(merged) >= 2:
        min_first = float(cfg["min_first_chunk_sec"])
        first_duration = merged[0].end_sec - merged[0].start_sec
        if first_duration < min_first:
            merged[1] = _merge_unit_pair(merged[0], merged[1], unit_id=0)
            merged = merged[1:]

    if len(merged) >= 2:
        min_last = float(cfg["min_last_chunk_sec"])
        last_duration = merged[-1].end_sec - merged[-1].start_sec
        if last_duration < min_last:
            merged[-2] = _merge_unit_pair(merged[-2], merged[-1], unit_id=len(merged) - 2)
            merged = merged[:-1]
    return merged


def _build_slide_embeddings(
    slide_keypoints: Sequence[Sequence[Keypoint]],
    model_name: str,
    model: Any | None = None,
) -> np.ndarray:
    model = model or _get_model(model_name)
    slide_vectors = []
    dimension = None
    for keypoints in slide_keypoints:
        if keypoints:
            texts = [kp.text for kp in keypoints]
            weights = np.array([kp.importance for kp in keypoints], dtype=float)
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            dimension = int(embs.shape[1])
            weighted = (embs.T * weights).T
            avg = weighted.sum(axis=0) / max(weights.sum(), 1e-8)
            norm = float(np.linalg.norm(avg))
            if norm > 0.0:
                avg = avg / norm
            slide_vectors.append(avg)
        else:
            if dimension is None:
                dimension = 768
            slide_vectors.append(np.zeros(dimension, dtype=float))
    return np.vstack(slide_vectors)


def _encode_weighted_average(texts: Sequence[str], weights: Sequence[float], *, model: Any, dimension: int) -> np.ndarray | None:
    clean_texts = [text.strip() for text in texts if text and text.strip()]
    if not clean_texts:
        return None
    clean_weights = np.array(list(weights)[: len(clean_texts)], dtype=float)
    if clean_weights.size != len(clean_texts):
        clean_weights = np.ones(len(clean_texts), dtype=float)
    embs = model.encode(clean_texts, convert_to_numpy=True, normalize_embeddings=True)
    avg = (embs.T * clean_weights).T.sum(axis=0) / max(float(clean_weights.sum()), 1e-8)
    norm = float(np.linalg.norm(avg))
    if norm > 0.0:
        avg = avg / norm
    if avg.shape[0] != dimension:
        return np.zeros(dimension, dtype=float)
    return avg


def _build_slide_representation(
    keypoints: Sequence[Keypoint],
    *,
    model: Any,
    dimension: int,
    config: Dict | None = None,
) -> SlideRepresentation:
    cfg = _alignment_config(config)
    titles = [kp for kp in keypoints if kp.source == "title" and kp.text.strip()]
    bodies = [kp for kp in keypoints if kp.source in {"bullet", "textrank"} and kp.text.strip()]
    keywords = [kp for kp in keypoints if kp.source in {"body", "vlm_entity", "vlm_keyword", "vlm_core_term"} and kp.text.strip()]
    visuals = [kp for kp in keypoints if kp.source == "visual" and kp.text.strip()]
    claims = [kp for kp in keypoints if kp.source == "vlm_claim" and kp.text.strip()]
    roles = [kp for kp in keypoints if kp.source == "vlm_role" and kp.text.strip()]
    vlm_visuals = [kp for kp in keypoints if kp.source == "vlm_visual" and kp.text.strip()]

    title_text = titles[0].text if titles else ""
    body_text = " ".join(kp.text for kp in bodies)
    keyword_texts = [kp.text for kp in keywords]
    visual_text = " ".join(kp.text for kp in visuals)
    vlm_visual_text = " ".join(kp.text for kp in vlm_visuals)

    title_emb = _encode_weighted_average([kp.text for kp in titles], [kp.importance for kp in titles], model=model, dimension=dimension)
    body_emb = _encode_weighted_average([kp.text for kp in bodies], [kp.importance for kp in bodies], model=model, dimension=dimension)
    keyword_emb = _encode_weighted_average(keyword_texts, [kp.importance for kp in keywords], model=model, dimension=dimension)
    visual_emb = _encode_weighted_average([visual_text], [1.0], model=model, dimension=dimension)
    claim_emb = _encode_weighted_average([kp.text for kp in claims], [kp.importance for kp in claims], model=model, dimension=dimension)
    role_emb = _encode_weighted_average([kp.text for kp in roles], [kp.importance for kp in roles], model=model, dimension=dimension)
    vlm_visual_emb = _encode_weighted_average([vlm_visual_text], [1.0], model=model, dimension=dimension)

    title_tokens = set(tokenize_nouns(title_text))
    all_keywords = set(title_tokens)
    all_keywords.update(token.lower() for token in keyword_texts if token.strip())
    all_keywords.update(tokenize_nouns(body_text))
    all_keywords.update(tokenize_nouns(visual_text))
    all_keywords.update(tokenize_nouns(vlm_visual_text))

    slide_type = None
    visual_kind = None
    for kp in keypoints:
        if kp.source == "vlm_slide_type" and kp.text.strip():
            slide_type = kp.text.strip()
        elif kp.source == "vlm_visual_kind" and kp.text.strip():
            visual_kind = kp.text.strip()
    has_vlm = bool(claims or roles or vlm_visuals or slide_type or visual_kind)
    is_visual_heavy = bool(visual_kind and visual_kind != "none")

    title_weight = float(cfg["short_title_weight"] if is_short_title(title_text, cfg) else cfg["default_title_weight"])
    return SlideRepresentation(
        title_emb=title_emb,
        body_emb=body_emb,
        keyword_emb=keyword_emb,
        visual_emb=visual_emb,
        claim_emb=claim_emb,
        role_emb=role_emb,
        vlm_visual_emb=vlm_visual_emb,
        title_weight=title_weight,
        title_tokens=title_tokens,
        all_keywords=all_keywords,
        slide_type=slide_type,
        visual_kind=visual_kind,
        has_vlm=has_vlm,
        is_visual_heavy=is_visual_heavy,
    )


def _cos_unit(unit_emb: np.ndarray, slide_emb: np.ndarray | None) -> float:
    if slide_emb is None:
        return 0.0
    return float(np.dot(unit_emb, slide_emb))


def _sim_utt_slide(unit_emb: np.ndarray, unit_text: str, slide_repr: SlideRepresentation, config: Dict | None = None) -> float:
    cfg = _alignment_config(config)
    s_title = _cos_unit(unit_emb, slide_repr.title_emb)
    s_body = _cos_unit(unit_emb, slide_repr.body_emb)
    s_kw = _cos_unit(unit_emb, slide_repr.keyword_emb)
    s_vis = _cos_unit(unit_emb, slide_repr.visual_emb)

    title_weight = slide_repr.title_weight if slide_repr.title_emb is not None else 0.0
    detail_weight = max(0.0, 1.0 - title_weight)
    pra_sem = title_weight * s_title + detail_weight * (
        float(cfg["body_semantic_weight"]) * s_body
        + float(cfg["keyword_semantic_weight"]) * s_kw
        + float(cfg["visual_semantic_weight"]) * s_vis
    )
    sem = pra_sem
    if slide_repr.has_vlm:
        s_claim = _cos_unit(unit_emb, slide_repr.claim_emb)
        s_role = _cos_unit(unit_emb, slide_repr.role_emb)
        s_vlm_vis = _cos_unit(unit_emb, slide_repr.vlm_visual_emb)
        if slide_repr.slide_type in {"title", "thanks", "section_header"}:
            vlm_sem = float(cfg["vlm_title_claim_weight"]) * s_claim + float(cfg["vlm_title_role_weight"]) * s_role
        elif slide_repr.is_visual_heavy:
            vlm_sem = (
                float(cfg["vlm_visual_claim_weight"]) * s_claim
                + float(cfg["vlm_visual_role_weight"]) * s_role
                + float(cfg["vlm_visual_visual_weight"]) * s_vlm_vis
            )
        else:
            vlm_sem = (
                float(cfg["vlm_default_claim_weight"]) * s_claim
                + float(cfg["vlm_default_role_weight"]) * s_role
                + float(cfg["vlm_default_visual_weight"]) * s_vlm_vis
            )
        vlm_mix = min(max(float(cfg["vlm_semantic_mix_weight"]), 0.0), 0.35)
        sem = (1.0 - vlm_mix) * pra_sem + vlm_mix * vlm_sem

    unit_nouns = set(tokenize_nouns(unit_text))
    lex = len(unit_nouns & slide_repr.all_keywords) / max(len(slide_repr.all_keywords), 1) if slide_repr.all_keywords else 0.0
    title_hit = 1.0 if unit_nouns & slide_repr.title_tokens else 0.0
    return (
        float(cfg["semantic_weight"]) * sem
        + float(cfg["lexical_weight"]) * lex
        + float(cfg["title_hit_weight"]) * title_hit
    )


def _build_multi_signal_similarity_matrix(
    unit_emb: np.ndarray,
    unit_texts: Sequence[str],
    slide_keypoints: Sequence[Sequence[Keypoint]],
    *,
    model: Any,
    config: Dict | None = None,
) -> np.ndarray:
    dimension = int(unit_emb.shape[1])
    slide_reprs = [
        _build_slide_representation(keypoints, model=model, dimension=dimension, config=config)
        for keypoints in slide_keypoints
    ]
    sim = np.zeros((len(unit_texts), len(slide_reprs)), dtype=float)
    for unit_idx, text in enumerate(unit_texts):
        for slide_idx, slide_repr in enumerate(slide_reprs):
            sim[unit_idx, slide_idx] = _sim_utt_slide(unit_emb[unit_idx], text, slide_repr, config=config)
    return sim


def _build_channel_embeddings(
    slide_keypoints: Sequence[Sequence[Keypoint]],
    *,
    model: Any,
    dimension: int,
    sources: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    active = []
    for keypoints in slide_keypoints:
        selected = [kp for kp in keypoints if kp.source in sources and kp.text.strip()]
        if not selected:
            vectors.append(np.zeros(dimension, dtype=float))
            active.append(False)
            continue
        texts = [kp.text for kp in selected]
        weights = np.array([kp.importance for kp in selected], dtype=float)
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        weighted = (embs.T * weights).T
        avg = weighted.sum(axis=0) / max(weights.sum(), 1e-8)
        norm = float(np.linalg.norm(avg))
        if norm > 0.0:
            avg = avg / norm
        vectors.append(avg)
        active.append(True)
    return np.vstack(vectors), np.array(active, dtype=bool)


def _build_alignment_similarity_matrix(
    unit_emb: np.ndarray,
    slide_keypoints: Sequence[Sequence[Keypoint]],
    *,
    model_name: str,
    model: Any,
    config: Dict | None = None,
) -> np.ndarray:
    cfg = _alignment_config(config)
    base_slide_emb = _build_slide_embeddings(slide_keypoints, model_name=model_name, model=model)
    base_sim = cosine_similarity(unit_emb, base_slide_emb)
    if not bool(cfg.get("use_multi_signal_similarity", True)):
        return base_sim
    # unit_texts is injected by auto_align_slides through config to preserve the
    # public helper signature used in tests and older callers.
    unit_texts = cfg.get("_unit_texts")
    if not unit_texts:
        return base_sim
    return _build_multi_signal_similarity_matrix(
        unit_emb,
        unit_texts,
        slide_keypoints,
        model=model,
        config=cfg,
    )


def _decode_monotonic_path(sim_matrix: np.ndarray, config: Dict | None = None) -> List[int]:
    return _decode_flexible_monotonic_path(sim_matrix, config=config)


def _emission_score(
    sim_matrix: np.ndarray,
    unit_idx: int,
    slide_idx: int,
    progress_penalty: float,
    lag_penalty_multiplier: float,
) -> float:
    sim = float(sim_matrix[unit_idx, slide_idx])
    n_units, n_slides = sim_matrix.shape
    if n_units <= 1 or n_slides <= 1:
        return sim
    unit_progress = unit_idx / max(1, n_units - 1)
    slide_progress = slide_idx / max(1, n_slides - 1)
    delta = slide_progress - unit_progress
    penalty_scale = lag_penalty_multiplier if delta < 0 else 1.0
    return sim - progress_penalty * abs(delta) * penalty_scale


def _decode_flexible_monotonic_path(sim_matrix: np.ndarray, config: Dict | None = None) -> List[int]:
    cfg = _alignment_config(config)
    n_units, n_slides = sim_matrix.shape
    if n_units == 0 or n_slides == 0:
        return []

    min_dwell_units = int(cfg.get("min_dwell_units", 1))
    if min_dwell_units > 1:
        return _decode_dwell_monotonic_path(sim_matrix, config=cfg, min_dwell_units=min_dwell_units)

    switch_penalty = float(cfg.get("switch_penalty", 0.08))
    skip_penalty = float(cfg.get("skip_penalty", 0.2))
    start_skip_penalty = float(cfg.get("start_skip_penalty", skip_penalty))
    allow_skip = bool(cfg.get("allow_skip", True))
    progress_penalty = float(cfg.get("progress_penalty", 0.35))
    lag_penalty_multiplier = float(cfg.get("lag_penalty_multiplier", 1.6))

    dp = np.full((n_units, n_slides), -np.inf, dtype=float)
    back = np.full((n_units, n_slides), -1, dtype=int)

    for slide_idx in range(n_slides):
        dp[0, slide_idx] = _emission_score(
            sim_matrix=sim_matrix,
            unit_idx=0,
            slide_idx=slide_idx,
            progress_penalty=progress_penalty,
            lag_penalty_multiplier=lag_penalty_multiplier,
        ) - start_skip_penalty * slide_idx

    for unit_idx in range(1, n_units):
        for slide_idx in range(n_slides):
            candidates = [(dp[unit_idx - 1, slide_idx], slide_idx)]
            if slide_idx > 0:
                candidates.append((dp[unit_idx - 1, slide_idx - 1] - switch_penalty, slide_idx - 1))
            if allow_skip and slide_idx > 1:
                candidates.append((dp[unit_idx - 1, slide_idx - 2] - skip_penalty, slide_idx - 2))
            best_prev, best_slide = max(candidates, key=lambda item: item[0])
            dp[unit_idx, slide_idx] = _emission_score(
                sim_matrix=sim_matrix,
                unit_idx=unit_idx,
                slide_idx=slide_idx,
                progress_penalty=progress_penalty,
                lag_penalty_multiplier=lag_penalty_multiplier,
            ) + best_prev
            back[unit_idx, slide_idx] = best_slide

    slide_idx = int(np.argmax(dp[-1]))
    assignments = [0] * n_units
    for unit_idx in range(n_units - 1, -1, -1):
        assignments[unit_idx] = slide_idx
        slide_idx = int(back[unit_idx, slide_idx]) if unit_idx > 0 else slide_idx
    return assignments


def _decode_dwell_monotonic_path(
    sim_matrix: np.ndarray,
    config: Dict | None = None,
    min_dwell_units: int = 2,
) -> List[int]:
    cfg = _alignment_config(config)
    n_units, n_slides = sim_matrix.shape
    if n_units == 0 or n_slides == 0:
        return []

    switch_penalty = float(cfg.get("switch_penalty", 0.08))
    skip_penalty = float(cfg.get("skip_penalty", 0.2))
    start_skip_penalty = float(cfg.get("start_skip_penalty", skip_penalty))
    allow_skip = bool(cfg.get("allow_skip", True))
    max_jump = int(cfg.get("max_slide_jump", 2 if allow_skip else 1))
    progress_penalty = float(cfg.get("progress_penalty", 0.35))
    lag_penalty_multiplier = float(cfg.get("lag_penalty_multiplier", 1.6))
    extra_skip_penalty = float(cfg.get("extra_skip_penalty", 0.4))
    stay_bonus = float(cfg.get("stay_bonus", 0.05))
    dwell_cap = max(1, min_dwell_units)

    dp = np.full((n_units, n_slides, dwell_cap + 1), -np.inf, dtype=float)
    back_slide = np.full((n_units, n_slides, dwell_cap + 1), -1, dtype=int)
    back_dwell = np.full((n_units, n_slides, dwell_cap + 1), -1, dtype=int)

    for slide_idx in range(n_slides):
        dp[0, slide_idx, 1] = _emission_score(
            sim_matrix=sim_matrix,
            unit_idx=0,
            slide_idx=slide_idx,
            progress_penalty=progress_penalty,
            lag_penalty_multiplier=lag_penalty_multiplier,
        ) - start_skip_penalty * slide_idx

    for unit_idx in range(1, n_units):
        for prev_slide in range(n_slides):
            for prev_dwell in range(1, dwell_cap + 1):
                prev_score = dp[unit_idx - 1, prev_slide, prev_dwell]
                if np.isneginf(prev_score):
                    continue

                # Stay on the same slide.
                next_dwell = min(dwell_cap, prev_dwell + 1)
                score = prev_score + stay_bonus + _emission_score(
                    sim_matrix=sim_matrix,
                    unit_idx=unit_idx,
                    slide_idx=prev_slide,
                    progress_penalty=progress_penalty,
                    lag_penalty_multiplier=lag_penalty_multiplier,
                )
                if score > dp[unit_idx, prev_slide, next_dwell]:
                    dp[unit_idx, prev_slide, next_dwell] = score
                    back_slide[unit_idx, prev_slide, next_dwell] = prev_slide
                    back_dwell[unit_idx, prev_slide, next_dwell] = prev_dwell

                if prev_dwell < dwell_cap:
                    continue
                jump_limit = min(max_jump, n_slides - 1 - prev_slide)
                for jump in range(1, jump_limit + 1):
                    next_slide = prev_slide + jump
                    jump_penalty = switch_penalty if jump == 1 else skip_penalty * jump + extra_skip_penalty * max(0, jump - 1)
                    score = prev_score - jump_penalty + _emission_score(
                        sim_matrix=sim_matrix,
                        unit_idx=unit_idx,
                        slide_idx=next_slide,
                        progress_penalty=progress_penalty,
                        lag_penalty_multiplier=lag_penalty_multiplier,
                    )
                    if score > dp[unit_idx, next_slide, 1]:
                        dp[unit_idx, next_slide, 1] = score
                        back_slide[unit_idx, next_slide, 1] = prev_slide
                        back_dwell[unit_idx, next_slide, 1] = prev_dwell

    flat_idx = int(np.argmax(dp[-1]))
    slide_idx, dwell_idx = np.unravel_index(flat_idx, dp[-1].shape)
    assignments = [0] * n_units
    for unit_idx in range(n_units - 1, -1, -1):
        assignments[unit_idx] = int(slide_idx)
        if unit_idx == 0:
            break
        prev_slide = int(back_slide[unit_idx, slide_idx, dwell_idx])
        prev_dwell = int(back_dwell[unit_idx, slide_idx, dwell_idx])
        if prev_slide < 0 or prev_dwell < 0:
            break
        slide_idx, dwell_idx = prev_slide, prev_dwell
    return assignments


def _decode_full_coverage_monotonic_path(sim_matrix: np.ndarray, config: Dict | None = None) -> List[int]:
    """Decode a conservative path that visits every slide in order.

    This is intentionally stricter than the primary DP and is used only as a
    low-confidence recovery path. It forbids skips and duplicate/empty slides
    when there are enough timed units to assign at least one unit per slide.
    """
    cfg = _alignment_config(config)
    n_units, n_slides = sim_matrix.shape
    if n_units == 0 or n_slides == 0:
        return []
    if n_slides == 1:
        return [0] * n_units
    if n_units < n_slides:
        return _decode_flexible_monotonic_path(sim_matrix, config=cfg)

    switch_penalty = float(cfg.get("switch_penalty", 0.08))
    progress_penalty = float(cfg.get("robust_progress_penalty", cfg.get("progress_penalty", 0.35)))
    lag_penalty_multiplier = float(cfg.get("lag_penalty_multiplier", 1.6))
    stay_bonus = float(cfg.get("robust_stay_bonus", 0.0))

    dp = np.full((n_units, n_slides), -np.inf, dtype=float)
    back = np.full((n_units, n_slides), -1, dtype=int)
    dp[0, 0] = _emission_score(
        sim_matrix=sim_matrix,
        unit_idx=0,
        slide_idx=0,
        progress_penalty=progress_penalty,
        lag_penalty_multiplier=lag_penalty_multiplier,
    )

    for unit_idx in range(1, n_units):
        for slide_idx in range(n_slides):
            if slide_idx > unit_idx:
                continue
            if (n_units - 1 - unit_idx) < (n_slides - 1 - slide_idx):
                continue

            emission = _emission_score(
                sim_matrix=sim_matrix,
                unit_idx=unit_idx,
                slide_idx=slide_idx,
                progress_penalty=progress_penalty,
                lag_penalty_multiplier=lag_penalty_multiplier,
            )
            stay_score = dp[unit_idx - 1, slide_idx] + stay_bonus
            best_score = stay_score
            best_prev = slide_idx
            if slide_idx > 0:
                advance_score = dp[unit_idx - 1, slide_idx - 1] - switch_penalty
                if advance_score > best_score:
                    best_score = advance_score
                    best_prev = slide_idx - 1
            if np.isneginf(best_score):
                continue
            dp[unit_idx, slide_idx] = best_score + emission
            back[unit_idx, slide_idx] = best_prev

    if np.isneginf(dp[-1, -1]):
        return _decode_flexible_monotonic_path(sim_matrix, config=cfg)

    slide_idx = n_slides - 1
    assignments = [0] * n_units
    for unit_idx in range(n_units - 1, -1, -1):
        assignments[unit_idx] = slide_idx
        if unit_idx == 0:
            break
        slide_idx = int(back[unit_idx, slide_idx])
        if slide_idx < 0:
            return _decode_flexible_monotonic_path(sim_matrix, config=cfg)
    return assignments


def _decode_segmental_full_coverage_path(
    sim_matrix: np.ndarray,
    units: Sequence[TimedUnit],
    config: Dict | None = None,
) -> List[int]:
    """Segment-level fallback DP that assigns every slide a contiguous run."""
    cfg = _alignment_config(config)
    n_units, n_slides = sim_matrix.shape
    if n_units == 0 or n_slides == 0:
        return []
    if n_slides == 1:
        return [0] * n_units
    if n_units < n_slides:
        return _decode_full_coverage_monotonic_path(sim_matrix, config=cfg)

    prefix = np.vstack([np.zeros((1, n_slides), dtype=float), np.cumsum(sim_matrix, axis=0)])
    audio_start = float(units[0].start_sec)
    audio_end = float(units[-1].end_sec)
    total_duration = max(audio_end - audio_start, 0.1)
    target_duration = total_duration / n_slides
    min_duration = max(
        float(cfg.get("robust_min_segment_sec", 12.0)),
        target_duration * float(cfg.get("robust_min_segment_ratio", 0.25)),
    )
    duration_weight = float(cfg.get("robust_duration_penalty_weight", 0.25))
    short_weight = float(cfg.get("robust_short_segment_penalty_weight", 1.2))
    progress_weight = float(cfg.get("robust_segment_progress_penalty", 0.35))
    pause_bonus = float(cfg.get("robust_boundary_pause_bonus", 0.08))
    pause_threshold = float(cfg.get("pause_threshold_sec", 0.9))

    def segment_score(start_idx: int, end_idx: int, slide_idx: int) -> float:
        sim_sum = float(prefix[end_idx, slide_idx] - prefix[start_idx, slide_idx])
        start_sec = float(units[start_idx].start_sec)
        end_sec = float(units[end_idx - 1].end_sec)
        duration = max(end_sec - start_sec, 0.1)
        duration_penalty = duration_weight * abs(np.log(duration / max(target_duration, 0.1)))
        if duration < min_duration:
            duration_penalty += short_weight * (min_duration - duration) / max(min_duration, 0.1)
        midpoint_progress = ((start_sec + end_sec) * 0.5 - audio_start) / total_duration
        slide_progress = (slide_idx + 0.5) / n_slides
        duration_penalty += progress_weight * abs(midpoint_progress - slide_progress)
        boundary_bonus = 0.0
        if start_idx > 0:
            gap = float(units[start_idx].start_sec - units[start_idx - 1].end_sec)
            if gap >= pause_threshold:
                boundary_bonus += pause_bonus
        return sim_sum - duration_penalty + boundary_bonus

    dp = np.full((n_slides, n_units + 1), -np.inf, dtype=float)
    back = np.full((n_slides, n_units + 1), -1, dtype=int)

    for end_idx in range(1, n_units - n_slides + 2):
        dp[0, end_idx] = segment_score(0, end_idx, 0)

    for slide_idx in range(1, n_slides):
        min_end = slide_idx + 1
        max_end = n_units - (n_slides - 1 - slide_idx)
        for end_idx in range(min_end, max_end + 1):
            best_score = -np.inf
            best_start = -1
            for start_idx in range(slide_idx, end_idx):
                prev_score = dp[slide_idx - 1, start_idx]
                if np.isneginf(prev_score):
                    continue
                score = prev_score + segment_score(start_idx, end_idx, slide_idx)
                if score > best_score:
                    best_score = score
                    best_start = start_idx
            dp[slide_idx, end_idx] = best_score
            back[slide_idx, end_idx] = best_start

    if np.isneginf(dp[-1, -1]):
        return _decode_full_coverage_monotonic_path(sim_matrix, config=cfg)

    segments: List[tuple[int, int, int]] = []
    end_idx = n_units
    for slide_idx in range(n_slides - 1, -1, -1):
        start_idx = int(back[slide_idx, end_idx]) if slide_idx > 0 else 0
        if start_idx < 0:
            return _decode_full_coverage_monotonic_path(sim_matrix, config=cfg)
        segments.append((start_idx, end_idx, slide_idx))
        end_idx = start_idx
    segments.reverse()

    assignments = [0] * n_units
    for start_idx, end_idx, slide_idx in segments:
        for unit_idx in range(start_idx, end_idx):
            assignments[unit_idx] = slide_idx
    return assignments


def _slide_has_any_text(keypoints: Sequence[Keypoint], patterns: Sequence[str]) -> bool:
    text = " ".join(kp.text for kp in keypoints).lower()
    return any(pattern.lower() in text for pattern in patterns)


def _slide_title_text(keypoints: Sequence[Keypoint]) -> str:
    for kp in keypoints:
        if kp.source == "title" and kp.text.strip():
            return kp.text
    return ""


def _unit_contains_title_tokens(unit: TimedUnit, title_tokens: set[str]) -> bool:
    if not title_tokens:
        return False
    unit_tokens = set(tokenize_nouns(unit.text))
    return bool(unit_tokens & title_tokens)


def _resolve_cover_end(
    units: Sequence[TimedUnit],
    slide_keypoints: Sequence[Sequence[Keypoint]] | None = None,
    config: Dict | None = None,
) -> float | None:
    cfg = _alignment_config(config)
    audio_end = float(units[-1].end_sec) if units else 0.0
    if audio_end <= 0.0:
        return None

    cover_window = max(
        _config_float(cfg, "cover_min_window", "cover_min_duration_sec"),
        min(
            _config_float(cfg, "cover_max_window", "cover_window_max_sec"),
            audio_end * _config_float(cfg, "cover_ratio", "cover_window_ratio"),
        ),
    )
    overrun_sec = max(0.0, float(cfg.get("cover_end_overrun_sec", 0.0)))
    cover_end_limit = cover_window + overrun_sec
    last_inside_end = None
    for unit in units:
        # A unit that starts inside the cover window and spills just over the
        # limit is usually still title/introduction speech. Large spills are
        # excluded so the cover anchor cannot drift to a full next section.
        if unit.end_sec <= cover_end_limit + 1e-6:
            last_inside_end = float(unit.end_sec)
            continue
        break
    if last_inside_end is None:
        return cover_window

    inside_units = [unit for unit in units if unit.end_sec <= cover_end_limit + 1e-6]
    if bool(cfg.get("cover_stop_on_next_title", True)) and slide_keypoints and len(slide_keypoints) >= 2:
        next_title_tokens = set(tokenize_nouns(_slide_title_text(slide_keypoints[1])))
        for idx, unit in enumerate(inside_units):
            if idx > 0 and _unit_contains_title_tokens(unit, next_title_tokens):
                return float(inside_units[idx - 1].end_sec)

    if bool(cfg.get("cover_prefer_first_unit", True)) and inside_units:
        first = inside_units[0]
        if first.end_sec - first.start_sec >= float(cfg.get("cover_first_unit_min_sec", 3.0)):
            return float(first.end_sec)

    return last_inside_end


def _resolve_thanks_start(
    units: Sequence[TimedUnit],
    slide_keypoints: Sequence[Sequence[Keypoint]],
    config: Dict | None = None,
) -> float | None:
    cfg = _alignment_config(config)
    if not units:
        return None
    audio_end = float(units[-1].end_sec)
    if not slide_keypoints or not _slide_has_any_text(slide_keypoints[-1], ["감사", "thank"]):
        return None

    closing_cues = tuple(cfg["closing_cues"])
    min_start = audio_end * float(cfg["thanks_search_start_ratio"])
    for unit in reversed(units):
        unit_text = unit.text.replace(" ", "")
        if any(str(cue).replace(" ", "") in unit_text for cue in closing_cues):
            return float(unit.start_sec) if unit.start_sec >= min_start else audio_end
    return audio_end


def _anchor_cover_and_thanks(
    units: Sequence[TimedUnit],
    slide_keypoints: Sequence[Sequence[Keypoint]],
    config: Dict | None = None,
) -> EdgeAnchors:
    cfg = _alignment_config(config)
    audio_end = float(units[-1].end_sec) if units else 0.0
    cover_end = 0.0
    cover_anchored = False
    if bool(cfg["anchor_cover_slide"]):
        cover_end = float(_resolve_cover_end(units, slide_keypoints, cfg) or 0.0)
        cover_anchored = cover_end > 0.0
    thanks_start = audio_end
    thanks_anchored = False
    if bool(cfg["anchor_thanks_slide"]):
        resolved = _resolve_thanks_start(units, slide_keypoints, cfg)
        if resolved is not None:
            thanks_start = resolved
            thanks_anchored = True
    return EdgeAnchors(
        cover_end_sec=cover_end,
        thanks_start_sec=thanks_start,
        cover_anchored=cover_anchored,
        thanks_anchored=thanks_anchored,
    )


def _apply_edge_anchors(
    units: Sequence[TimedUnit],
    assignments: Sequence[int],
    slide_keypoints: Sequence[Sequence[Keypoint]],
    config: Dict | None = None,
) -> List[int]:
    cfg = config or {}
    if not units or not assignments:
        return list(assignments)

    anchored = list(assignments)
    slide_count = len(slide_keypoints)
    if slide_count <= 1:
        return anchored

    audio_end = float(units[-1].end_sec)

    if bool(cfg.get("anchor_cover_slide", True)):
        cover_end = _resolve_cover_end(units, slide_keypoints, cfg)
        if cover_end is not None:
            for idx, unit in enumerate(units):
                if unit.end_sec <= cover_end + 1e-6:
                    anchored[idx] = 0
                elif anchored[idx] == 0:
                    anchored[idx] = 1

    closing_cues = tuple(cfg.get("closing_cues", ["감사", "들어주셔", "이상입니다", "마치겠습니다"]))
    if bool(cfg.get("anchor_thanks_slide", True)) and _slide_has_any_text(slide_keypoints[-1], ["감사", "thank"]):
        last_slide = slide_count - 1
        search_start = audio_end * float(cfg.get("thanks_search_start_ratio", 0.75))
        thanks_idx = None
        for idx in range(len(units) - 1, -1, -1):
            unit_text = units[idx].text.replace(" ", "")
            if units[idx].start_sec >= search_start and any(cue.replace(" ", "") in unit_text for cue in closing_cues):
                thanks_idx = idx
                break
        if thanks_idx is None:
            anchored = [min(item, last_slide - 1) for item in anchored]
        else:
            for idx in range(thanks_idx, len(anchored)):
                anchored[idx] = last_slide
            for idx in range(0, thanks_idx):
                if anchored[idx] >= last_slide:
                    anchored[idx] = last_slide - 1

    for idx in range(1, len(anchored)):
        anchored[idx] = max(anchored[idx], anchored[idx - 1])
        anchored[idx] = min(anchored[idx], slide_count - 1)
    return anchored


def _build_boundaries(
    units: Sequence[TimedUnit],
    assignments: Sequence[int],
    slide_count: int,
) -> List[float]:
    if slide_count <= 0:
        return []
    if not units:
        return [0.0] * (slide_count + 1)

    end_sec = float(units[-1].end_sec)
    boundaries = [0.0]
    for slide_idx in range(1, slide_count):
        start_sec = end_sec
        for unit, assigned in zip(units, assignments):
            if assigned >= slide_idx:
                start_sec = float(unit.start_sec)
                break
        boundaries.append(max(boundaries[-1], start_sec))
    boundaries.append(max(boundaries[-1], end_sec))
    return boundaries


def _ordered_unique_tokens(tokens: Sequence[str]) -> List[str]:
    ordered = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return ordered


def _find_title_cue_start(unit: TimedUnit, title_tokens: Sequence[str]) -> float | None:
    if not title_tokens:
        return None
    required = _ordered_unique_tokens(title_tokens)[:2]
    if not required:
        return None
    words = list(unit.words)
    for idx, word in enumerate(words):
        current_tokens = set(tokenize_nouns(_word_text(word)))
        if required[0] not in current_tokens:
            continue
        window_tokens = set()
        for next_word in words[idx : idx + 6]:
            window_tokens.update(tokenize_nouns(_word_text(next_word)))
        if all(token in window_tokens for token in required):
            return _word_start(word)
    return None


def _find_first_title_token_start(unit: TimedUnit, title_tokens: Sequence[str]) -> float | None:
    required = _ordered_unique_tokens(title_tokens)
    if not required:
        return None
    first_token = required[0]
    for word in unit.words:
        if first_token in set(tokenize_nouns(_word_text(word))):
            return _word_start(word)
    return None


def _refine_boundaries_with_title_cues(
    boundaries: Sequence[float],
    units: Sequence[TimedUnit],
    assignments: Sequence[int],
    slide_keypoints: Sequence[Sequence[Keypoint]],
    config: Dict | None = None,
) -> List[float]:
    cfg = _alignment_config(config)
    if not bool(cfg["title_cue_boundary_refinement_enabled"]) or not units or not assignments:
        return list(boundaries)

    refined = list(boundaries)
    min_offset = float(cfg["title_cue_refine_min_offset_sec"])
    search_max_units = int(cfg["title_cue_search_max_units"])
    slide_count = len(slide_keypoints)
    last_refinable_exclusive = slide_count if slide_count <= 2 else slide_count - 1
    for slide_idx in range(1, last_refinable_exclusive):
        title_text = _slide_title_text(slide_keypoints[slide_idx])
        title_tokens = tokenize_nouns(title_text)
        if not title_tokens:
            continue
        boundary_unit_idx = None
        for unit_idx, assigned in enumerate(assignments):
            if assigned >= slide_idx:
                boundary_unit_idx = unit_idx
                break
        if boundary_unit_idx is None:
            continue
        boundary_unit = units[boundary_unit_idx]
        cue_start = None
        run_end = boundary_unit_idx
        while run_end < len(assignments) and assignments[run_end] == assignments[boundary_unit_idx]:
            run_end += 1
        for unit in units[boundary_unit_idx : min(run_end, boundary_unit_idx + search_max_units)]:
            cue_start = _find_title_cue_start(unit, title_tokens)
            if cue_start is not None:
                break
        allow_first_token_fallback = bool(cfg["title_cue_allow_first_token_fallback"]) and "소개" in title_text
        if cue_start is None and allow_first_token_fallback:
            for unit in units[boundary_unit_idx : min(run_end, boundary_unit_idx + search_max_units)]:
                cue_start = _find_first_title_token_start(unit, title_tokens)
                if cue_start is not None:
                    break
        if cue_start is None:
            continue
        if cue_start - boundary_unit.start_sec < min_offset:
            continue
        lower = refined[slide_idx - 1]
        upper = refined[slide_idx + 1] if slide_idx + 1 < len(refined) else refined[-1]
        refined[slide_idx] = min(max(cue_start, lower), upper)

    for idx in range(1, len(refined)):
        refined[idx] = max(refined[idx], refined[idx - 1])
    return refined


def _assignment_runs(assignments: Sequence[int]) -> List[tuple[int, int, int]]:
    runs: List[tuple[int, int, int]] = []
    if not assignments:
        return runs
    start = 0
    current = assignments[0]
    for idx, slide_idx in enumerate(assignments[1:], start=1):
        if slide_idx != current:
            runs.append((start, idx, current))
            start = idx
            current = slide_idx
    runs.append((start, len(assignments), current))
    return runs


def _mean_margin_for_indices(sim_matrix: np.ndarray, assignments: Sequence[int], indices: Sequence[int]) -> float:
    margins = []
    for idx in indices:
        row = sim_matrix[idx]
        assigned = int(assignments[idx])
        assigned_score = float(row[assigned])
        if len(row) <= 1:
            margins.append(1.0)
            continue
        competitor = float(np.max(np.delete(row, assigned)))
        margins.append(assigned_score - competitor)
    return float(np.mean(margins)) if margins else 0.0


def _mean_score_for_indices(sim_matrix: np.ndarray, indices: Sequence[int], slide_idx: int) -> float:
    if not indices:
        return 0.0
    return float(np.mean([sim_matrix[idx, slide_idx] for idx in indices]))


def _run_duration(units: Sequence[TimedUnit], start: int, end: int) -> float:
    if start >= end:
        return 0.0
    return float(units[end - 1].end_sec - units[start].start_sec)


def _best_neighbor_for_indices(
    sim_matrix: np.ndarray,
    indices: Sequence[int],
    candidates: Sequence[int],
) -> int | None:
    if not candidates:
        return None
    return int(max(candidates, key=lambda slide_idx: _mean_score_for_indices(sim_matrix, indices, slide_idx)))


def _absorb_pause_chunks(
    assignments: Sequence[int],
    units: Sequence[TimedUnit],
    sim_matrix: np.ndarray,
    config: Dict | None = None,
    fixed_slides: set[int] | None = None,
) -> List[int]:
    cfg = _alignment_config(config)
    if not bool(cfg["absorb_pause_chunks_enabled"]) or not assignments:
        return list(assignments)

    absorbed = list(assignments)
    fixed = fixed_slides or set()
    min_chunk_sec = float(cfg["absorb_min_chunk_sec"])
    edge_max_sec = float(cfg["absorb_edge_max_sec"])
    max_utts = int(cfg["absorb_max_chunk_utts"])
    margin_threshold = float(cfg["absorb_margin_threshold"])
    runs = _assignment_runs(absorbed)

    # First, absorb a whole short weak run when it is clearly better explained
    # by one of its non-anchored neighbors.
    for run_idx, (start, end, slide_idx) in enumerate(runs):
        indices = list(range(start, end))
        if slide_idx in fixed or len(indices) > max_utts or _run_duration(units, start, end) >= min_chunk_sec:
            continue
        if _mean_margin_for_indices(sim_matrix, absorbed, indices) >= margin_threshold:
            continue
        candidates = []
        if run_idx > 0 and runs[run_idx - 1][2] not in fixed:
            candidates.append(runs[run_idx - 1][2])
        if run_idx + 1 < len(runs) and runs[run_idx + 1][2] not in fixed:
            candidates.append(runs[run_idx + 1][2])
        best_slide = _best_neighbor_for_indices(sim_matrix, indices, candidates)
        if best_slide is None:
            continue
        current_score = _mean_score_for_indices(sim_matrix, indices, slide_idx)
        if _mean_score_for_indices(sim_matrix, indices, best_slide) <= current_score:
            continue
        for idx in indices:
            absorbed[idx] = best_slide

    runs = _assignment_runs(absorbed)

    # Then handle the common pause failure mode: the first one or two units of a
    # new run are weak and belong semantically to the previous slide, while the
    # rest of the run is correct. This keeps DP intact and only moves boundary
    # edge crumbs.
    for run_idx in range(1, len(runs)):
        prev_start, prev_end, prev_slide = runs[run_idx - 1]
        start, end, slide_idx = runs[run_idx]
        if prev_slide in fixed or slide_idx in fixed:
            continue

        max_prefix_end = min(end, start + max_utts)
        best_prefix: List[int] = []
        for prefix_end in range(start + 1, max_prefix_end + 1):
            candidate = list(range(start, prefix_end))
            if _run_duration(units, start, prefix_end) <= edge_max_sec:
                best_prefix = candidate
        if best_prefix:
            margin = _mean_margin_for_indices(sim_matrix, absorbed, best_prefix)
            prev_score = _mean_score_for_indices(sim_matrix, best_prefix, prev_slide)
            current_score = _mean_score_for_indices(sim_matrix, best_prefix, slide_idx)
            if margin < margin_threshold and prev_score > current_score:
                for idx in best_prefix:
                    absorbed[idx] = prev_slide

        max_suffix_start = max(prev_start, prev_end - max_utts)
        best_suffix: List[int] = []
        for suffix_start in range(prev_end - 1, max_suffix_start - 1, -1):
            candidate = list(range(suffix_start, prev_end))
            if _run_duration(units, suffix_start, prev_end) <= edge_max_sec:
                best_suffix = candidate
        if best_suffix:
            margin = _mean_margin_for_indices(sim_matrix, absorbed, best_suffix)
            next_score = _mean_score_for_indices(sim_matrix, best_suffix, slide_idx)
            current_score = _mean_score_for_indices(sim_matrix, best_suffix, prev_slide)
            if margin < margin_threshold and next_score > current_score:
                for idx in best_suffix:
                    absorbed[idx] = slide_idx

    for idx in range(1, len(absorbed)):
        absorbed[idx] = max(absorbed[idx], absorbed[idx - 1])
    return absorbed


def _smooth_low_confidence_runs(
    assignments: Sequence[int],
    sim_matrix: np.ndarray,
    config: Dict | None = None,
    fixed_slides: set[int] | None = None,
) -> List[int]:
    cfg = _alignment_config(config)
    if not bool(cfg["smoothing_enabled"]) or not assignments:
        return list(assignments)

    smoothed = list(assignments)
    fixed = fixed_slides or set()
    threshold = float(cfg["smoothing_margin_threshold"])
    max_units = int(cfg["smoothing_max_units"])
    runs = _assignment_runs(smoothed)

    for run_idx, (start, end, slide_idx) in enumerate(runs):
        if slide_idx in fixed or (end - start) > max_units:
            continue
        if bool(cfg["smoothing_preserve_assigned_slides"]) and smoothed.count(slide_idx) == (end - start):
            continue
        indices = list(range(start, end))
        if _mean_margin_for_indices(sim_matrix, smoothed, indices) >= threshold:
            continue

        candidates = []
        if run_idx > 0:
            candidates.append(runs[run_idx - 1][2])
        if run_idx + 1 < len(runs):
            candidates.append(runs[run_idx + 1][2])
        candidates = [candidate for candidate in candidates if candidate not in fixed or candidate == slide_idx]
        if not candidates:
            continue

        best_slide = max(
            candidates,
            key=lambda candidate: float(np.mean([sim_matrix[idx, candidate] for idx in indices])),
        )
        for idx in indices:
            smoothed[idx] = int(best_slide)
    for idx in range(1, len(smoothed)):
        smoothed[idx] = max(smoothed[idx], smoothed[idx - 1])
    return smoothed


def _collect_alignment_details(
    units: Sequence[TimedUnit],
    assignments: Sequence[int],
    confidence_sim_matrix: np.ndarray,
    config: Dict | None = None,
) -> tuple[List[Dict], List[Dict], float | None]:
    cfg = _alignment_config(config)
    min_similarity = float(cfg["confidence_min_similarity"])
    min_margin = float(cfg["confidence_min_margin"])
    low_confidence_segments: List[Dict] = []
    unit_assignments: List[Dict] = []
    confidence_values: List[float] = []

    for unit_idx, unit in enumerate(units):
        row = confidence_sim_matrix[unit_idx]
        assigned = int(assignments[unit_idx])
        best_score = float(row[assigned])
        if len(row) > 1:
            competitor = float(np.max(np.delete(row, assigned)))
            margin = best_score - competitor
        else:
            margin = 1.0
        confidence_values.append(max(0.0, min(1.0, margin)))
        payload = {
            "unit_id": unit.unit_id,
            "slide_id": assigned + 1,
            "start_sec": unit.start_sec,
            "end_sec": unit.end_sec,
            "text": unit.text,
            "similarity": round(best_score, 4),
            "margin": round(margin, 4),
        }
        unit_assignments.append(payload)
        if best_score < min_similarity or margin < min_margin:
            reason = []
            if best_score < min_similarity:
                reason.append("low_similarity")
            if margin < min_margin:
                reason.append("ambiguous_margin")
            low_confidence_segments.append({**payload, "reason": ",".join(reason)})

    confidence = float(np.mean(confidence_values)) if confidence_values else None
    return unit_assignments, low_confidence_segments, confidence


def _alignment_warnings(
    assignments: Sequence[int],
    boundaries: Sequence[float],
    slide_count: int,
    confidence: float | None,
    config: Dict | None = None,
) -> List[str]:
    cfg = _alignment_config(config)
    warnings: List[str] = []
    covered_slides = {assignment + 1 for assignment in assignments}
    if len(covered_slides) < slide_count:
        warnings.append("not_all_slides_assigned")
    if len(set(boundaries)) < len(boundaries):
        warnings.append("duplicate_boundaries_detected")
    low_threshold = float(cfg.get("low_confidence_warning_threshold", 0.0))
    if confidence is not None and low_threshold > 0.0 and confidence < low_threshold:
        warnings.append("low_alignment_confidence")
    return warnings


def _should_apply_robust_fallback(warnings: Sequence[str], confidence: float | None, config: Dict | None = None) -> bool:
    cfg = _alignment_config(config)
    if not bool(cfg.get("robust_fallback_enabled", True)):
        return False
    if bool(cfg.get("robust_fallback_on_duplicate", True)) and "duplicate_boundaries_detected" in warnings:
        return True
    if bool(cfg.get("robust_fallback_on_unassigned", True)) and "not_all_slides_assigned" in warnings:
        return True
    threshold = float(cfg.get("robust_fallback_confidence_threshold", 0.0))
    return confidence is not None and threshold > 0.0 and confidence < threshold


def _apply_robust_fallback(
    assignments: Sequence[int],
    units: Sequence[TimedUnit],
    sim_matrix: np.ndarray,
    slide_count: int,
    anchors: EdgeAnchors,
    config: Dict | None = None,
) -> List[int] | None:
    if not units or slide_count <= 0:
        return None
    cfg = _alignment_config(config)
    min_units_per_slide = max(1, int(cfg.get("robust_fallback_min_units_per_slide", 1)))

    last_slide_idx = slide_count - 1
    first_dp_slide = 1 if anchors.cover_anchored and slide_count > 1 else 0
    last_dp_exclusive = last_slide_idx if anchors.thanks_anchored and slide_count > 1 else slide_count
    middle_slide_indices = list(range(first_dp_slide, last_dp_exclusive))
    middle_unit_indices = [
        idx
        for idx, unit in enumerate(units)
        if (not anchors.cover_anchored or unit.start_sec >= anchors.cover_end_sec)
        and (not anchors.thanks_anchored or unit.start_sec < anchors.thanks_start_sec)
    ]
    if not middle_slide_indices:
        return None
    if len(middle_unit_indices) < len(middle_slide_indices) * min_units_per_slide:
        return None

    robust = list(assignments)
    for idx, unit in enumerate(units):
        if anchors.cover_anchored and unit.end_sec <= anchors.cover_end_sec + 1e-6:
            robust[idx] = 0
        elif anchors.thanks_anchored and unit.start_sec >= anchors.thanks_start_sec:
            robust[idx] = last_slide_idx

    sub_sim = sim_matrix[np.ix_(middle_unit_indices, middle_slide_indices)]
    sub_units = [units[idx] for idx in middle_unit_indices]
    if bool(cfg.get("robust_segmental_dp_enabled", True)):
        sub_assignments = _decode_segmental_full_coverage_path(sub_sim, sub_units, config=cfg)
    else:
        sub_assignments = _decode_full_coverage_monotonic_path(sub_sim, config=cfg)
    if len(set(sub_assignments)) < len(middle_slide_indices):
        return None
    for local_unit_idx, local_slide_idx in zip(middle_unit_indices, sub_assignments):
        robust[local_unit_idx] = middle_slide_indices[int(local_slide_idx)]

    for idx in range(1, len(robust)):
        robust[idx] = max(robust[idx], robust[idx - 1])
        robust[idx] = min(robust[idx], slide_count - 1)
    return robust


def auto_align_slides(
    slide_keypoints: Sequence[Sequence[Keypoint]],
    words: Sequence[Dict],
    model_name: str = "jhgan/ko-sroberta-multitask",
    config: Dict | None = None,
) -> AlignmentResult:
    cfg = _alignment_config(config)
    slide_count = len(slide_keypoints)
    warnings: List[str] = []
    if slide_count <= 0:
        return AlignmentResult(mode="auto", strategy_used="auto", final_boundaries=[], proposed_boundaries=[])
    if not words:
        zeros = [0.0] * (slide_count + 1)
        return AlignmentResult(
            mode="auto",
            strategy_used="auto",
            final_boundaries=zeros,
            proposed_boundaries=zeros,
            warnings=["no_words_for_auto_alignment"],
        )

    units = build_timed_units(words, config=cfg)
    if not units:
        zeros = [0.0] * (slide_count + 1)
        return AlignmentResult(
            mode="auto",
            strategy_used="auto",
            final_boundaries=zeros,
            proposed_boundaries=zeros,
            warnings=["failed_to_build_timed_units"],
        )

    model = _get_model(model_name)
    unit_texts = [unit.text for unit in units]
    unit_emb = model.encode(unit_texts, convert_to_numpy=True, normalize_embeddings=True)
    sim_cfg = dict(cfg)
    sim_cfg["_unit_texts"] = unit_texts
    has_vlm = _has_vlm_keypoints(slide_keypoints)
    boundary_keypoints = (
        slide_keypoints
        if bool(cfg.get("vlm_boundary_influence_enabled", False)) or not has_vlm
        else _without_vlm_keypoints(slide_keypoints)
    )
    sim_matrix = _build_alignment_similarity_matrix(
        unit_emb,
        boundary_keypoints,
        model_name=model_name,
        model=model,
        config=sim_cfg,
    )
    confidence_sim_matrix = sim_matrix
    if has_vlm and boundary_keypoints is not slide_keypoints:
        confidence_sim_matrix = _build_alignment_similarity_matrix(
            unit_emb,
            slide_keypoints,
            model_name=model_name,
            model=model,
            config=sim_cfg,
        )
    assignments = [0] * len(units)
    anchors = _anchor_cover_and_thanks(units, slide_keypoints, cfg)
    last_slide_idx = slide_count - 1

    if slide_count == 1:
        assignments = [0] * len(units)
    elif slide_count == 2:
        if anchors.thanks_anchored:
            for idx, unit in enumerate(units):
                assignments[idx] = last_slide_idx if unit.start_sec >= anchors.thanks_start_sec else 0
        elif anchors.cover_anchored:
            for idx, unit in enumerate(units):
                assignments[idx] = 0 if unit.end_sec <= anchors.cover_end_sec + 1e-6 else last_slide_idx
        else:
            assignments = _decode_monotonic_path(sim_matrix, config=cfg)
    else:
        first_dp_slide = 1 if anchors.cover_anchored else 0
        last_dp_exclusive = last_slide_idx if anchors.thanks_anchored else slide_count
        middle_slide_indices = list(range(first_dp_slide, last_dp_exclusive))
        middle_unit_indices = [
            idx
            for idx, unit in enumerate(units)
            if (not anchors.cover_anchored or unit.start_sec >= anchors.cover_end_sec)
            and (not anchors.thanks_anchored or unit.start_sec < anchors.thanks_start_sec)
        ]
        for idx, unit in enumerate(units):
            if anchors.thanks_anchored and unit.start_sec >= anchors.thanks_start_sec:
                assignments[idx] = last_slide_idx
            elif anchors.cover_anchored and unit.end_sec <= anchors.cover_end_sec + 1e-6:
                assignments[idx] = 0
            else:
                assignments[idx] = middle_slide_indices[0] if middle_slide_indices else 0

        if middle_unit_indices and middle_slide_indices:
            sub_sim = sim_matrix[np.ix_(middle_unit_indices, middle_slide_indices)]
            sub_assignments = _decode_monotonic_path(sub_sim, config=cfg)
            for local_idx, slide_offset in zip(middle_unit_indices, sub_assignments):
                assignments[local_idx] = middle_slide_indices[int(slide_offset)]

    fixed_slides = set()
    if anchors.cover_anchored:
        fixed_slides.add(0)
    if anchors.thanks_anchored:
        fixed_slides.add(last_slide_idx)
    assignments = _absorb_pause_chunks(
        assignments,
        units,
        sim_matrix,
        config=cfg,
        fixed_slides=fixed_slides,
    )
    assignments = _smooth_low_confidence_runs(
        assignments,
        sim_matrix,
        config=cfg,
        fixed_slides=fixed_slides,
    )
    boundaries = _build_boundaries(units, assignments, slide_count)
    boundaries = _refine_boundaries_with_title_cues(
        boundaries,
        units,
        assignments,
        boundary_keypoints,
        config=cfg,
    )

    unit_assignments, low_confidence_segments, confidence = _collect_alignment_details(
        units,
        assignments,
        confidence_sim_matrix,
        config=cfg,
    )
    warnings = _alignment_warnings(assignments, boundaries, slide_count, confidence, config=cfg)
    if _should_apply_robust_fallback(warnings, confidence, config=cfg):
        robust_assignments = _apply_robust_fallback(
            assignments,
            units,
            sim_matrix,
            slide_count,
            anchors,
            config=cfg,
        )
        if robust_assignments is not None and robust_assignments != assignments:
            assignments = robust_assignments
            boundaries = _build_boundaries(units, assignments, slide_count)
            boundaries = _refine_boundaries_with_title_cues(
                boundaries,
                units,
                assignments,
                boundary_keypoints,
                config=cfg,
            )
            unit_assignments, low_confidence_segments, confidence = _collect_alignment_details(
                units,
                assignments,
                confidence_sim_matrix,
                config=cfg,
            )
            warnings = ["robust_fallback_applied"] + _alignment_warnings(
                assignments,
                boundaries,
                slide_count,
                confidence,
                config=cfg,
            )
        else:
            warnings = ["robust_fallback_unavailable"] + warnings

    return AlignmentResult(
        mode="auto",
        strategy_used="auto_robust_fallback" if "robust_fallback_applied" in warnings else "auto",
        final_boundaries=boundaries,
        proposed_boundaries=boundaries,
        confidence=confidence,
        low_confidence_segments=low_confidence_segments,
        unit_assignments=unit_assignments,
        warnings=warnings,
    )


def resolve_alignment(
    slide_keypoints: Sequence[Sequence[Keypoint]],
    words: Sequence[Dict],
    model_name: str,
    mode: str | None = None,
    provided_boundaries: Sequence[float] | None = None,
    config: Dict | None = None,
) -> AlignmentResult:
    requested_mode = (mode or "").strip().lower()
    if not requested_mode:
        requested_mode = "manual" if provided_boundaries else "auto"
    if requested_mode not in {"manual", "auto", "hybrid"}:
        raise ValueError(f"Unsupported alignment mode: {requested_mode}")

    manual_boundaries: List[float] = []
    manual_warnings: List[str] = []
    manual_error: ValueError | None = None

    if provided_boundaries:
        try:
            manual_boundaries, manual_warnings = normalize_manual_boundaries(
                provided_boundaries=provided_boundaries,
                slide_count=len(slide_keypoints),
                words=words,
            )
        except ValueError as exc:
            manual_error = exc

    if requested_mode == "manual":
        if manual_error is not None:
            raise manual_error
        if not manual_boundaries:
            raise ValueError("Manual alignment mode requires slide_timestamps.")
        return AlignmentResult(
            mode="manual",
            strategy_used="manual",
            final_boundaries=manual_boundaries,
            provided_boundaries=manual_boundaries,
            warnings=manual_warnings,
        )

    auto_result = auto_align_slides(
        slide_keypoints=slide_keypoints,
        words=words,
        model_name=model_name,
        config=config,
    )

    if requested_mode == "auto":
        auto_result.mode = "auto"
        auto_result.provided_boundaries = manual_boundaries
        auto_result.warnings = manual_warnings + auto_result.warnings
        return auto_result

    if manual_error is not None:
        auto_result.mode = "hybrid"
        auto_result.strategy_used = "auto_after_invalid_manual"
        auto_result.warnings = [str(manual_error)] + auto_result.warnings
        return auto_result

    if manual_boundaries:
        return AlignmentResult(
            mode="hybrid",
            strategy_used="manual_with_auto_proposal",
            final_boundaries=manual_boundaries,
            provided_boundaries=manual_boundaries,
            proposed_boundaries=auto_result.final_boundaries,
            confidence=auto_result.confidence,
            low_confidence_segments=auto_result.low_confidence_segments,
            unit_assignments=auto_result.unit_assignments,
            warnings=manual_warnings + auto_result.warnings,
        )

    auto_result.mode = "hybrid"
    if auto_result.strategy_used == "auto":
        auto_result.strategy_used = "auto_without_manual_override"
    return auto_result
