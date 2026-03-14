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

    def analyze(
        self,
        document_path: str,
        audio_path: str,
        slide_timestamps: Sequence[float],
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

        done = self._time("transcript_alignment")
        segments = transcript_aligner.align_transcript(words, slide_timestamps)
        done()

        done = self._time("ce_scoring")
        ce_results = []
        for keypoints, segment in zip(slide_keypoints, segments):
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
        required=True,
        help="Comma-separated slide boundaries in seconds, e.g. 0,30,60,90",
    )
    parser.add_argument(
        "--whisper-json",
        default=None,
        help="Path to JSON with {'words': [{'word','start','end'}, ...]}. Optional if STT enabled.",
    )
    args = parser.parse_args()

    slide_timestamps = [float(x.strip()) for x in args.slide_timestamps.split(",") if x.strip()]
    whisper_result = None
    if args.whisper_json:
        whisper_result = json.loads(Path(args.whisper_json).read_text())

    pipeline = SpeechPTPipeline(config_path=args.config)
    report = pipeline.analyze(
        document_path=args.document,
        audio_path=args.audio,
        slide_timestamps=slide_timestamps,
        whisper_result=whisper_result,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
