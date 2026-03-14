import numpy as np

from speechpt.coherence.coherence_scorer import score_slide
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


def test_score_slide_covers_matching_keypoints(monkeypatch):
    monkeypatch.setattr("speechpt.coherence.coherence_scorer.SentenceTransformer", lambda *args, **kwargs: DummyModel())

    kps = [Keypoint(text="hello world", importance=1.0, source="title"), Keypoint(text="extra", importance=0.5, source="body")]
    transcript = TranscriptSegment(slide_id=1, start_sec=0.0, end_sec=5.0, text="hello world mentioned", words=[])

    result = score_slide(kps, transcript, threshold=0.1)
    # DummyModel maps length to embeddings so both keypoints should be covered.
    assert result.coverage > 0.9
    assert result.missed_keypoints == []
