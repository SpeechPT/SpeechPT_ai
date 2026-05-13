import numpy as np

from speechpt.coherence import coherence_scorer
from speechpt.coherence.keypoint_extractor import Keypoint
from speechpt.coherence.transcript_aligner import TranscriptSegment


class DummyModel:
    def __init__(self, *_, **__):
        pass

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # map text length to vector for deterministic similarity
        vecs = []
        for t in texts:
            val = len(t)
            vecs.append(np.array([val, 0, 0], dtype=float))
        arr = np.vstack(vecs)
        if normalize_embeddings:
            norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr = arr / norm
        return arr


class TokenMatchDummyModel:
    def __init__(self, *_, **__):
        pass

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vocab = ["title", "bullet", "visual", "tiny"]
        vecs = []
        for text in texts:
            lowered = text.lower()
            vec = np.array([1.0 if token in lowered else 0.0 for token in vocab], dtype=float)
            if vec.sum() == 0:
                vec[0] = 0.5
            vecs.append(vec)
        arr = np.vstack(vecs)
        if normalize_embeddings:
            norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr = arr / norm
        return arr


def test_score_slide_covers_matching_keypoints(monkeypatch):
    coherence_scorer._model_cache.clear()
    monkeypatch.setattr("speechpt.coherence.coherence_scorer.SentenceTransformer", lambda *args, **kwargs: DummyModel())

    kps = [Keypoint(text="hello world", importance=1.0, source="title"), Keypoint(text="extra", importance=0.5, source="body")]
    transcript = TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="hello world mentioned", words=[])

    result = coherence_scorer.score_slide(kps, transcript, threshold=0.1)
    # DummyModel maps length to embeddings so both keypoints should be covered.
    assert result.coverage > 0.9
    assert result.keypoint_coverage > 0.9
    assert result.semantic_coverage > 0.9
    assert result.soft_keypoint_coverage > 0.9
    assert result.missed_keypoints == []
    assert result.source_coverage["title"] > 0.9
    assert result.source_coverage["body"] > 0.9
    assert result.source_missed_keypoints["title"] == []


def test_score_slide_tracks_missed_keypoints_by_source(monkeypatch):
    coherence_scorer._model_cache.clear()
    monkeypatch.setattr("speechpt.coherence.coherence_scorer.SentenceTransformer", lambda *args, **kwargs: TokenMatchDummyModel())

    kps = [
        Keypoint(text="title only phrase", importance=1.0, source="title"),
        Keypoint(text="tiny", importance=0.8, source="bullet"),
        Keypoint(text="visual only phrase", importance=0.7, source="visual"),
    ]
    transcript = TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="tiny", words=[])

    result = coherence_scorer.score_slide(kps, transcript, threshold=0.8)
    assert result.keypoint_coverage < 0.5
    assert result.coverage < 0.6
    assert "title only phrase" in result.missed_keypoints
    assert "visual only phrase" in result.missed_keypoints
    assert "title only phrase" in result.source_missed_keypoints["title"]
    assert "visual only phrase" in result.source_missed_keypoints["visual"]


def test_score_slide_keeps_hard_keypoint_coverage_separate(monkeypatch):
    coherence_scorer._model_cache.clear()
    monkeypatch.setattr("speechpt.coherence.coherence_scorer.SentenceTransformer", lambda *args, **kwargs: TokenMatchDummyModel())

    kps = [
        Keypoint(text="title only phrase", importance=1.0, source="title"),
        Keypoint(text="tiny", importance=1.0, source="bullet"),
    ]
    transcript = TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="tiny", words=[])

    result = coherence_scorer.score_slide(kps, transcript, threshold=0.8)

    assert result.keypoint_coverage < result.coverage
    assert 0.0 <= result.semantic_coverage <= 1.0
    assert 0.0 <= result.soft_keypoint_coverage <= 1.0
