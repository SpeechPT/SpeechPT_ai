"""슬라이드 단위 키포인트 추출기.

규칙 기반 + TextRank 혼합으로 핵심 주장 문장을 반환한다.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List

import kss
import numpy as np
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Keypoint:
    text: str
    importance: float
    source: str  # "title" | "bullet" | "body" | "textrank"


def _normalize(text: str) -> str:
    return text.strip()


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in seq:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def textrank_sentences(sentences: List[str], top_k: int = 3, damping: float = 0.85, max_iter: int = 50) -> List[str]:
    if not sentences:
        return []
    vectorizer = TfidfVectorizer().fit(sentences)
    tfidf = vectorizer.transform(sentences)
    sim = (tfidf * tfidf.T).toarray()
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True) + 1e-8
    weight = sim / row_sums
    scores = np.ones(len(sentences)) / len(sentences)
    for _ in range(max_iter):
        scores = (1 - damping) + damping * weight.T.dot(scores)
    ranked = np.argsort(scores)[::-1]
    return [sentences[i] for i in ranked[:top_k]]


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    kiwi = Kiwi()
    nouns = [token.form for token in kiwi.tokenize(text) if token.tag.startswith("N")]
    freq = {}
    for n in nouns:
        freq[n] = freq.get(n, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def extract_keypoints(slide) -> List[Keypoint]:  # slide: SlideContent-like
    keypoints: List[Keypoint] = []

    if slide.title:
        keypoints.append(Keypoint(text=_normalize(slide.title), importance=1.0, source="title"))

    for bp in slide.bullet_points:
        norm = _normalize(bp)
        if norm:
            keypoints.append(Keypoint(text=norm, importance=0.8, source="bullet"))

    sentences = [s.strip() for s in kss.split_sentences(slide.text) if s.strip()]
    textrank_top = textrank_sentences(sentences, top_k=min(3, len(sentences)))
    for sent in textrank_top:
        keypoints.append(Keypoint(text=_normalize(sent), importance=0.6, source="textrank"))

    keywords = extract_keywords(slide.text)
    for kw in keywords:
        keypoints.append(Keypoint(text=kw, importance=0.4, source="body"))

    # 중복 제거 (텍스트 기준)
    deduped = []
    seen = set()
    for kp in keypoints:
        k = kp.text.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(kp)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Extract keypoints from a plain text slide body")
    parser.add_argument("text", type=str, help="Slide text")
    args = parser.parse_args()

    slide_mock = type("Slide", (), {"title": "", "bullet_points": [], "text": args.text})
    kps = extract_keypoints(slide_mock)
    for kp in kps:
        logger.info("[%s] %.2f %s", kp.source, kp.importance, kp.text)


if __name__ == "__main__":
    main()
