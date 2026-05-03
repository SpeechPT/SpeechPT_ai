"""SpeechPT 전체 오케스트레이션 파이프라인."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Sequence

import yaml

from speechpt.attitude.attitude_scorer import score_attitude
from speechpt.attitude.audio_feature_extractor import extract_audio_features
from speechpt.attitude.change_point_detector import detect_change_points
from speechpt.attitude.wav2vec2_embedder import Wav2Vec2Embedder
from speechpt.coherence import coherence_scorer, document_parser, keypoint_extractor, transcript_aligner
from speechpt.coherence.keypoint_extractor import Keypoint
from speechpt.coherence.visual_captioner import build_visual_captions
from speechpt.coherence.visual_ocr import enrich_slides_with_visual_ocr
from speechpt.report.report_generator import SpeechReport, generate_report
from speechpt.stt import transcribe_audio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SpeechPTPipeline:
    def __init__(self, config_path: str, device: str = "cpu"):
        self.config_path = Path(config_path)
        self.device = device
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.ce_cfg = self.cfg.get("coherence", {})
        self.ae_cfg = self.cfg.get("attitude", {})
        self.stt_cfg = self.cfg.get("stt", {})
        self.report_tpl = Path(self.cfg.get("report", {}).get("template", "speechpt/report/templates/feedback_ko.yaml"))

    def apply_runtime_overrides(self, overrides: Dict | None = None) -> None:
        if not overrides:
            return
        stt_overrides = overrides.get("stt", {})
        for key, value in stt_overrides.items():
            if value is not None:
                self.stt_cfg[key] = value

    def _time(self, desc: str):
        start = time.perf_counter()

        def end():
            elapsed = time.perf_counter() - start
            logger.info("[stage=%s] elapsed_sec=%.3f", desc, elapsed)

        return end

    def _resolve_whisper_result(self, audio_path: str, whisper_result: Dict | None) -> Dict:
        if whisper_result is not None and "words" in whisper_result:
            return whisper_result

        if not self.stt_cfg.get("enabled", False):
            raise ValueError(
                "whisper_result['words'] is required when STT is disabled. "
                "Provide --whisper-json or enable stt.enabled in config."
            )

        done = self._time("stt_transcription")
        result = transcribe_audio(audio_path, self.stt_cfg)
        done()
        if "words" not in result:
            raise ValueError("STT output does not include 'words'.")
        return result

    def _build_slide_keypoints(self, slide) -> list[Keypoint]:
        keypoints = keypoint_extractor.extract_keypoints(slide)
        for cap in slide.visual_captions:
            keypoints.append(Keypoint(text=f"VISUAL: {cap}", importance=0.7, source="visual"))
        # text dedupe preserving highest importance
        best = {}
        for kp in keypoints:
            k = kp.text.lower()
            if k not in best or kp.importance > best[k].importance:
                best[k] = kp
        return list(best.values())

    def _auto_align_slides(self, slides, slide_keypoints, words) -> list[float]:
        """STT 트랜스크립트와 슬라이드 키포인트 간의 시맨틱 유사도를 분석하여 자동으로 슬라이드 경계를 추정한다."""
        if not slides or not words:
            return [0.0] * len(slides)

        # 1. 트랜스크립트를 약 10초 단위의 청크로 분할
        duration = 10.0
        chunks = []
        current_chunk = []
        current_start = float(words[0].get("start", 0.0)) if words else 0.0
        
        for w in words:
            w_start = float(w.get("start", 0.0))
            if current_chunk and (w_start - current_start) >= duration:
                chunks.append({
                    "start": float(current_chunk[0].get("start", 0.0)),
                    "end": float(current_chunk[-1].get("end", 0.0)),
                    "text": " ".join([str(cw.get("word", "")) for cw in current_chunk])
                })
                current_chunk = [w]
                current_start = w_start
            else:
                current_chunk.append(w)
                
        if current_chunk:
            chunks.append({
                "start": float(current_chunk[0].get("start", 0.0)),
                "end": float(current_chunk[-1].get("end", 0.0)),
                "text": " ".join([str(cw.get("word", "")) for cw in current_chunk])
            })

        if not chunks:
            return [0.0] * len(slides)

        # 2. 임베딩 계산 (jhgan/ko-sroberta-multitask 모델 활용)
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from speechpt.coherence.coherence_scorer import _get_model
        
        model_name = self.ce_cfg.get("model_name", "jhgan/ko-sroberta-multitask")
        model = _get_model(model_name)

        chunk_texts = [c["text"] for c in chunks]
        chunk_emb = model.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True)

        slide_embs = []
        for kps in slide_keypoints:
            if kps:
                kp_texts = [kp.text for kp in kps]
                kp_imp = np.array([kp.importance for kp in kps], dtype=float)
                embs = model.encode(kp_texts, convert_to_numpy=True, normalize_embeddings=True)
                weighted = (embs.T * kp_imp).T
                avg_emb = weighted.sum(axis=0) / (kp_imp.sum() + 1e-8)
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    avg_emb /= norm
                slide_embs.append(avg_emb)
            else:
                slide_embs.append(np.zeros(chunk_emb.shape[1]))
                
        slide_emb = np.vstack(slide_embs)
        
        # 3. 유사도 행렬 계산 (Chunks x Slides)
        S = cosine_similarity(chunk_emb, slide_emb)

        # 4. 동적 계획법(DP)을 이용한 최적 단조 정렬 경로 탐색
        N = len(chunks)
        M = len(slides)
        
        dp = np.full((N, M), -np.inf)
        backptr = np.zeros((N, M), dtype=int)

        for j in range(M):
            dp[0, j] = S[0, j]

        for i in range(1, N):
            for j in range(M):
                best_k = 0
                best_val = -np.inf
                for k in range(j + 1):
                    # 슬라이드를 건너뛸 때 약간의 패널티 부여 (순차 진행 장려)
                    penalty = 0.0
                    if k < j:
                        penalty = -0.1 * (j - k)
                    val = dp[i-1, k] + penalty
                    if val > best_val:
                        best_val = val
                        best_k = k
                dp[i, j] = S[i, j] + best_val
                backptr[i, j] = best_k

        # 최적 경로 역추적 (Backtracking)
        j = int(np.argmax(dp[N-1]))
        assignments = [0] * N
        for i in range(N - 1, -1, -1):
            assignments[i] = j
            j = backptr[i, j]

        # 5. 각 슬라이드의 시작 타임스탬프 추출
        timestamps = [0.0] * M
        for k in range(1, M):
            first_idx = -1
            for i in range(N):
                if assignments[i] >= k:
                    first_idx = i
                    break
            if first_idx != -1:
                timestamps[k] = chunks[first_idx]["start"]
            else:
                timestamps[k] = chunks[-1]["end"]

        return timestamps

    def analyze(
        self,
        document_path: str,
        audio_path: str,
        slide_timestamps: Sequence[float] | None = None,
        whisper_result: Dict | None = None,
    ) -> SpeechReport:
        whisper_result = self._resolve_whisper_result(audio_path, whisper_result)
        words = whisper_result["words"]

        done = self._time("document_parsing")
        slides = document_parser.parse_document(document_path)
        done()

        visual_cfg = self.ce_cfg.get("visual", {})
        if visual_cfg.get("enabled", False):
            done = self._time("visual_ocr")
            enrich_slides_with_visual_ocr(slides, document_path, visual_cfg)
            min_conf = float(visual_cfg.get("min_confidence", 0.3))
            max_len = int(visual_cfg.get("max_text_len", 80))
            for slide in slides:
                slide.visual_captions = build_visual_captions(slide.visual_items, min_confidence=min_conf, max_text_len=max_len)
            done()

        done = self._time("keypoint_extraction")
        slide_keypoints = [self._build_slide_keypoints(slide) for slide in slides]
        done()

        # 자동 정렬 수행
        if not slide_timestamps:
            done = self._time("auto_slide_alignment")
            slide_timestamps = self._auto_align_slides(slides, slide_keypoints, words)
            logger.info("Auto-aligned slide timestamps: %s", [round(t, 2) for t in slide_timestamps])
            done()

        done = self._time("transcript_alignment")
        segments = transcript_aligner.align_transcript(words, slide_timestamps)
        done()

        done = self._time("ce_scoring")
        ce_results = []
        segment_map = {segment.slide_id: segment for segment in segments}
        for slide, keypoints in zip(slides, slide_keypoints):
            segment = segment_map.get(
                slide.slide_id,
                transcript_aligner.TranscriptSegment(
                    slide_id=slide.slide_id,
                    start_sec=0.0,
                    end_sec=0.0,
                    text="",
                    words=[],
                    warning_flags=["missing_segment"],
                ),
            )
            ce_results.append(
                coherence_scorer.score_slide(
                    keypoints,
                    segment,
                    model_name=self.ce_cfg.get("model_name", "jhgan/ko-sroberta-multitask"),
                    threshold=self.ce_cfg.get("threshold", 0.55),
                )
            )
        done()

        done = self._time("ae_feature_extraction")
        audio_feats = extract_audio_features(audio_path, words=words, config=self.ae_cfg)
        done()

        slide_segments = []
        times = list(slide_timestamps)
        if len(times) < len(slides) + 1:
            times.append(audio_feats.duration_sec)
        for i, slide in enumerate(slides):
            start = times[i]
            end = times[i + 1] if i + 1 < len(times) else audio_feats.duration_sec
            slide_segments.append({"slide_id": slide.slide_id, "start_sec": start, "end_sec": end})

        done = self._time("change_point_detection")
        cp_list = detect_change_points(
            {
                "energy": audio_feats.energy,
                "pitch": audio_feats.pitch,
                "speech_rate_per_sec": audio_feats.speech_rate_per_sec,
            },
            audio_feats.frame_times,
            config=self.ae_cfg,
        )
        done()

        wav2vec_embeddings = None
        wav2vec_times = None
        wav2vec_cfg = self.ae_cfg.get("wav2vec2", {})
        if wav2vec_cfg.get("use_probe", False):
            done = self._time("wav2vec_embedding")
            embedder = Wav2Vec2Embedder(
                model_name=wav2vec_cfg.get("model_name", "facebook/wav2vec2-base"),
                chunk_duration_sec=float(wav2vec_cfg.get("chunk_duration_sec", 30)),
            )
            wav2vec_embeddings, wav2vec_times = embedder.encode_with_times(
                audio_path,
                sample_rate=int(self.ae_cfg.get("audio", {}).get("sample_rate", 16000)),
            )
            done()

        done = self._time("ae_scoring")
        ae_results = score_attitude(
            audio_feats,
            slide_segments,
            change_points=cp_list,
            words=words,
            config=self.ae_cfg,
            wav2vec_embeddings=wav2vec_embeddings,
            wav2vec_times=wav2vec_times,
        )
        done()

        done = self._time("report_generation")
        report = generate_report(ce_results, ae_results, template_path=self.report_tpl, version=self.cfg.get("version", "0.3.0"))
        done()

        return report


__all__ = ["SpeechPTPipeline"]


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end SpeechPT pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline config yaml")
    parser.add_argument("--document", required=True, help="Path to PDF/PPT document")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--slide-timestamps",
        default=None,
        help="Comma-separated slide boundaries in seconds, e.g. 0,30,60,90. If omitted, auto-alignment is used.",
    )
    parser.add_argument(
        "--whisper-json",
        default=None,
        help="Path to JSON with {'words': [{'word','start','end'}, ...]}. Optional if STT enabled.",
    )
    parser.add_argument("--stt-enabled", action="store_true", help="Force-enable STT even if config disables it.")
    parser.add_argument("--stt-backend", default=None, choices=["faster-whisper", "openai-whisper"])
    parser.add_argument("--stt-model", default=None, help="Override STT model name")
    parser.add_argument("--stt-language", default=None, help="Override STT language")
    parser.add_argument("--stt-compute-type", default=None, help="Override STT compute type")
    parser.add_argument("--stt-device", default=None, help="Override STT device")
    parser.add_argument("--save-whisper-json", default=None, help="Optional path to save the STT result JSON")
    args = parser.parse_args()

    slide_timestamps = None
    if args.slide_timestamps:
        slide_timestamps = [float(x.strip()) for x in args.slide_timestamps.split(",") if x.strip()]

    whisper_result = None
    if args.whisper_json:
        whisper_result = json.loads(Path(args.whisper_json).read_text())

    pipeline = SpeechPTPipeline(config_path=args.config)
    pipeline.apply_runtime_overrides(
        {
            "stt": {
                "enabled": True if args.stt_enabled else None,
                "backend": args.stt_backend,
                "model_name": args.stt_model,
                "language": args.stt_language,
                "compute_type": args.stt_compute_type,
                "device": args.stt_device,
            }
        }
    )
    if whisper_result is None and args.save_whisper_json:
        whisper_result = pipeline._resolve_whisper_result(args.audio, whisper_result=None)
        out = Path(args.save_whisper_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(whisper_result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = pipeline.analyze(
        document_path=args.document,
        audio_path=args.audio,
        slide_timestamps=slide_timestamps,
        whisper_result=whisper_result,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
