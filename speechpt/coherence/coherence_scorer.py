"""슬라이드별 정합성(Coherence) 스코어러."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import kss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .keypoint_extractor import Keypoint
from .transcript_aligner import TranscriptSegment

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_model_cache: Dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


@dataclass
class SlideCoherenceResult:
    slide_id: int
    coverage: float
    missed_keypoints: List[str]
    evidence_spans: List[Dict]
    source_coverage: Dict[str, float] | None = None
    source_missed_keypoints: Dict[str, List[str]] | None = None
    semantic_coverage: float | None = None
    soft_keypoint_coverage: float | None = None
    keypoint_coverage: float | None = None
    transcript_presence: float | None = None
    scoring_version: str = "v2"


def _chunk_transcript(text: str, max_len: int = 120) -> List[str]:
    sentences = [s.strip() for s in kss.split_sentences(text) if s.strip()]
    if not sentences:
        return []
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if len(current) + len(s) <= max_len:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


def _rescale_similarity(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _weighted_mean_embedding(embeddings: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0,), dtype=float)
    if weights is None:
        vector = np.mean(embeddings, axis=0)
    else:
        safe_weights = np.maximum(weights.astype(float), 0.0)
        if float(np.sum(safe_weights)) <= 1e-8:
            vector = np.mean(embeddings, axis=0)
        else:
            vector = np.average(embeddings, axis=0, weights=safe_weights)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def _weighted_score(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights)) + 1e-8
    return float(np.sum(values * weights) / total)


def score_slide(
    keypoints: List[Keypoint],
    transcript: TranscriptSegment,
    model_name: str = "jhgan/ko-sroberta-multitask",
    threshold: float = 0.55,
    scoring_config: Dict | None = None,
) -> SlideCoherenceResult:
    cfg = scoring_config or {}
    semantic_low = float(cfg.get("semantic_low", 0.40))
    semantic_high = float(cfg.get("semantic_high", 0.80))
    keypoint_soft_low = float(cfg.get("keypoint_soft_low", 0.35))
    keypoint_soft_high = float(cfg.get("keypoint_soft_high", 0.75))
    semantic_weight = float(cfg.get("semantic_weight", 0.45))
    soft_keypoint_weight = float(cfg.get("soft_keypoint_weight", 0.45))
    presence_weight = float(cfg.get("presence_weight", 0.10))
    min_transcript_chars = float(cfg.get("min_transcript_chars", 30.0))
    single_signal_cap = float(cfg.get("single_signal_cap", 0.72))
    agreement_min_score = float(cfg.get("agreement_min_score", 0.55))

    if not keypoints:
        return SlideCoherenceResult(
            slide_id=transcript.slide_id,
            coverage=0.0,
            missed_keypoints=[],
            evidence_spans=[],
            semantic_coverage=0.0,
            soft_keypoint_coverage=0.0,
            keypoint_coverage=0.0,
            transcript_presence=0.0,
        )

    model = _get_model(model_name)
    kp_texts = [kp.text for kp in keypoints]
    kp_importance = np.array([kp.importance for kp in keypoints], dtype=float)
    kp_emb = model.encode(kp_texts, convert_to_numpy=True, normalize_embeddings=True)

    chunks = _chunk_transcript(transcript.text)
    if not chunks:
        return SlideCoherenceResult(
            slide_id=transcript.slide_id,
            coverage=0.0,
            missed_keypoints=kp_texts,
            evidence_spans=[],
            semantic_coverage=0.0,
            soft_keypoint_coverage=0.0,
            keypoint_coverage=0.0,
            transcript_presence=0.0,
        )
    ch_emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    sim_matrix = cosine_similarity(kp_emb, ch_emb)
    max_sim = sim_matrix.max(axis=1)
    covered_mask = max_sim >= threshold

    covered_importance = kp_importance[covered_mask].sum()
    total_importance = kp_importance.sum() + 1e-8
    keypoint_coverage = float(covered_importance / total_importance)
    soft_scores = np.array([_rescale_similarity(float(score), keypoint_soft_low, keypoint_soft_high) for score in max_sim])
    soft_keypoint_coverage = _weighted_score(soft_scores, kp_importance)

    slide_vector = _weighted_mean_embedding(kp_emb, kp_importance)
    transcript_vector = _weighted_mean_embedding(ch_emb)
    raw_semantic = float(np.dot(slide_vector, transcript_vector)) if slide_vector.size and transcript_vector.size else 0.0
    semantic_coverage = _rescale_similarity(raw_semantic, semantic_low, semantic_high)
    transcript_presence = float(np.clip(len(transcript.text.strip()) / max(min_transcript_chars, 1.0), 0.0, 1.0))

    total_weight = semantic_weight + soft_keypoint_weight + presence_weight
    if total_weight <= 1e-8:
        coverage = keypoint_coverage
    else:
        coverage = (
            semantic_weight * semantic_coverage
            + soft_keypoint_weight * soft_keypoint_coverage
            + presence_weight * transcript_presence
        ) / total_weight
    if min(semantic_coverage, soft_keypoint_coverage) < agreement_min_score:
        coverage = min(float(coverage), single_signal_cap)
    coverage = float(np.clip(coverage, 0.0, 1.0))

    source_coverage: Dict[str, float] = {}
    source_missed_keypoints: Dict[str, List[str]] = {}
    sources = sorted({kp.source for kp in keypoints})
    for source in sources:
        source_indices = np.array([i for i, kp in enumerate(keypoints) if kp.source == source], dtype=int)
        source_total = float(kp_importance[source_indices].sum()) + 1e-8
        source_covered = float(kp_importance[source_indices][covered_mask[source_indices]].sum())
        source_coverage[source] = float(source_covered / source_total)
        source_missed_keypoints[source] = [kp_texts[int(i)] for i in source_indices if not covered_mask[int(i)]]

    evidence_spans: List[Dict] = []
    for idx, kp in enumerate(keypoints):
        best_idx = int(sim_matrix[idx].argmax())
        if sim_matrix[idx, best_idx] >= threshold:
            evidence_spans.append(
                {
                    "keypoint": kp.text,
                    "transcript_chunk": chunks[best_idx],
                    "similarity": float(sim_matrix[idx, best_idx]),
                }
            )

    missed = [kp_texts[i] for i, covered in enumerate(covered_mask) if not covered]

    return SlideCoherenceResult(
        slide_id=transcript.slide_id,
        coverage=coverage,
        missed_keypoints=missed,
        evidence_spans=evidence_spans,
        source_coverage=source_coverage,
        source_missed_keypoints=source_missed_keypoints,
        semantic_coverage=semantic_coverage,
        soft_keypoint_coverage=soft_keypoint_coverage,
        keypoint_coverage=keypoint_coverage,
        transcript_presence=transcript_presence,
    )


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Score coherence for a slide")
    parser.add_argument("config", type=str, help="Path to coherence config yaml")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    threshold = cfg.get("threshold", 0.55)
    model_name = cfg.get("model_name", "jhgan/ko-sroberta-multitask")

    # demo data
    keypoints = [
        Keypoint(text="모델 아키텍처 소개", importance=1.0, source="title"),
        Keypoint(text="데이터 수집", importance=0.8, source="bullet"),
    ]
    transcript = TranscriptSegment(slide_id=1, start_sec=0, end_sec=30, text="모델 아키텍처를 설명하고 데이터 수집 과정을 공유했습니다.", words=[])
    result = score_slide(keypoints, transcript, model_name=model_name, threshold=threshold)
    logger.info("coverage=%.2f missed=%s", result.coverage, result.missed_keypoints)


if __name__ == "__main__":
    main()
