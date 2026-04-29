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


def score_slide(
    keypoints: List[Keypoint],
    transcript: TranscriptSegment,
    model_name: str = "jhgan/ko-sroberta-multitask",
    threshold: float = 0.55,
) -> SlideCoherenceResult:
    if not keypoints:
        return SlideCoherenceResult(
            slide_id=transcript.slide_id,
            coverage=0.0,
            missed_keypoints=[],
            evidence_spans=[],
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
        )
    ch_emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    sim_matrix = cosine_similarity(kp_emb, ch_emb)
    max_sim = sim_matrix.max(axis=1)
    covered_mask = max_sim >= threshold

    covered_importance = kp_importance[covered_mask].sum()
    total_importance = kp_importance.sum() + 1e-8
    coverage = float(covered_importance / total_importance)

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
