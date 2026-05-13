"""SpeechPT 전체 오케스트레이션 파이프라인."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Sequence

import yaml

from speechpt.attitude.attitude_scorer import score_attitude
from speechpt.attitude.ae_probe_runtime import predict_segments as predict_ae_probe_segments
from speechpt.attitude.audio_feature_extractor import extract_audio_features
from speechpt.attitude.change_point_detector import detect_change_points
from speechpt.attitude.wav2vec2_embedder import Wav2Vec2Embedder
from speechpt.coherence import auto_aligner, coherence_scorer, document_parser, keypoint_extractor, slide_role_classifier, transcript_aligner, vlm_caption
from speechpt.coherence.keypoint_extractor import Keypoint
from speechpt.coherence.visual_captioner import build_visual_captions
from speechpt.coherence.visual_ocr import enrich_slides_with_visual_ocr
from speechpt.report.llm_writer import build_llm_feedback
from speechpt.report.report_generator import SpeechReport, generate_report
from speechpt.stt import transcribe_audio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _serialize_transcript_segments(segments: Sequence[transcript_aligner.TranscriptSegment]) -> list[Dict]:
    return [
        {
            "slide_id": segment.slide_id,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "text": segment.text,
            "words": segment.words,
            "warning_flags": segment.warning_flags,
        }
        for segment in segments
    ]


def _call_suppressing_stdout(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    noise = buffer.getvalue().strip()
    if noise:
        logger.debug("Suppressed stdout from %s: %s", getattr(func, "__name__", repr(func)), noise)
    return result


class SpeechPTPipeline:
    def __init__(self, config_path: str, device: str = "cpu"):
        self.config_path = Path(config_path)
        self.device = device
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.ce_cfg = self.cfg.get("coherence", {})
        self.ae_cfg = self.cfg.get("attitude", {})
        self.stt_cfg = self.cfg.get("stt", {})
        self.report_cfg = self.cfg.get("report", {})
        self.report_tpl = Path(self.report_cfg.get("template", "speechpt/report/templates/feedback_ko.yaml"))

    def apply_runtime_overrides(self, overrides: Dict | None = None) -> None:
        if not overrides:
            return
        stt_overrides = overrides.get("stt", {})
        for key, value in stt_overrides.items():
            if value is not None:
                self.stt_cfg[key] = value
        vlm_overrides = overrides.get("vlm_caption", {})
        if vlm_overrides:
            vlm_cfg = self.ce_cfg.setdefault("vlm_caption", {})
            for key, value in vlm_overrides.items():
                if value is not None:
                    vlm_cfg[key] = value

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

    def _build_alignment_keypoints(
        self,
        slide_keypoints: Sequence[Sequence[Keypoint]],
        vlm_captions: Sequence[vlm_caption.VlmSlideCaption],
        vlm_presentation: vlm_caption.VlmPresentationCaption | None = None,
    ) -> list[list[Keypoint]]:
        caption_map = {caption.slide_id: caption for caption in vlm_captions}
        core_terms = vlm_presentation.core_terminology if vlm_presentation is not None else []
        alignment_keypoints: list[list[Keypoint]] = []
        for slide_idx, keypoints in enumerate(slide_keypoints, start=1):
            enhanced = list(keypoints)
            caption = caption_map.get(slide_idx)
            if caption is not None:
                enhanced.append(Keypoint(text=caption.slide_type, importance=0.25, source="vlm_slide_type"))
                enhanced.append(Keypoint(text=caption.visual_kind, importance=0.2, source="vlm_visual_kind"))
                if caption.role_in_flow:
                    enhanced.append(Keypoint(text=caption.role_in_flow, importance=0.75, source="vlm_role"))
                if caption.main_claim:
                    enhanced.append(Keypoint(text=caption.main_claim, importance=0.95, source="vlm_claim"))
                if caption.visual_summary:
                    enhanced.append(Keypoint(text=caption.visual_summary, importance=0.75, source="vlm_visual"))
                for entity in caption.entities:
                    enhanced.append(Keypoint(text=entity, importance=0.45, source="vlm_entity"))
                for keyword in caption.likely_keywords_in_speech:
                    enhanced.append(Keypoint(text=keyword, importance=0.45, source="vlm_keyword"))
            for term in core_terms:
                enhanced.append(Keypoint(text=term, importance=0.35, source="vlm_core_term"))

            best: dict[tuple[str, str], Keypoint] = {}
            for kp in enhanced:
                key = (kp.source, kp.text.lower())
                if key not in best or kp.importance > best[key].importance:
                    best[key] = kp
            alignment_keypoints.append(list(best.values()))
        return alignment_keypoints

    def analyze(
        self,
        document_path: str,
        audio_path: str,
        slide_timestamps: Sequence[float] | None = None,
        whisper_result: Dict | None = None,
        alignment_mode: str | None = None,
    ) -> SpeechReport:
        whisper_result = self._resolve_whisper_result(audio_path, whisper_result)
        words = whisper_result["words"]

        done = self._time("document_parsing")
        slides = _call_suppressing_stdout(document_parser.parse_document, document_path)
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

        vlm_captions = []
        vlm_presentation = None
        vlm_cfg = self.ce_cfg.get("vlm_caption", {})
        if vlm_cfg.get("enabled", False):
            done = self._time("vlm_caption")
            vlm_result = vlm_caption.caption_document(document_path, slides, vlm_cfg)
            if vlm_result is not None:
                vlm_captions = vlm_result.slides
                vlm_presentation = vlm_result.presentation
            done()
        slide_roles = slide_role_classifier.classify_slide_roles(slides, vlm_captions)
        alignment_keypoints = self._build_alignment_keypoints(slide_keypoints, vlm_captions, vlm_presentation)

        requested_alignment_mode = alignment_mode or self.ce_cfg.get("alignment", {}).get("mode")
        if not requested_alignment_mode:
            requested_alignment_mode = "manual" if slide_timestamps else "auto"

        done = self._time("slide_alignment")
        alignment = auto_aligner.resolve_alignment(
            slide_keypoints=alignment_keypoints,
            words=words,
            model_name=self.ce_cfg.get("model_name", "jhgan/ko-sroberta-multitask"),
            mode=requested_alignment_mode,
            provided_boundaries=slide_timestamps,
            config=self.ce_cfg.get("alignment", {}),
        )
        logger.info(
            "Resolved slide alignment mode=%s strategy=%s boundaries=%s",
            alignment.mode,
            alignment.strategy_used,
            [round(t, 2) for t in alignment.final_boundaries],
        )
        done()

        done = self._time("transcript_alignment")
        segments = transcript_aligner.align_transcript(words, alignment.final_boundaries)
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
                    scoring_config=self.ce_cfg.get("scoring_v2", {}),
                )
            )
        done()

        done = self._time("ae_feature_extraction")
        audio_feats = extract_audio_features(audio_path, words=words, config=self.ae_cfg)
        done()

        slide_segments = []
        times = list(alignment.final_boundaries)
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

        ae_probe_cfg = self.ae_cfg.get("ae_probe", {})
        if ae_probe_cfg.get("enabled", False):
            done = self._time("ae_probe_inference")
            try:
                ae_probe_predictions = predict_ae_probe_segments(audio_path, slide_segments, ae_probe_cfg)
                predictions_by_slide = {item.slide_id: item for item in ae_probe_predictions}
                for result in ae_results:
                    prediction = predictions_by_slide.get(result.slide_id)
                    if prediction is not None:
                        result.features.update(prediction.to_feature_dict())
            except Exception:
                if ae_probe_cfg.get("fail_open", True):
                    logger.exception("AE probe inference failed; continuing without ae_probe features")
                else:
                    raise
            finally:
                done()

        done = self._time("report_generation")
        report = generate_report(
            ce_results,
            ae_results,
            template_path=self.report_tpl,
            version=self.cfg.get("version", "0.3.0"),
            alignment=alignment.to_dict(),
            attitude_config=self.ae_cfg,
            report_config=self.cfg.get("report", {}),
            transcript_segments=_serialize_transcript_segments(segments),
            slide_roles=slide_roles,
        )
        done()

        llm_cfg = self.report_cfg.get("llm", {})
        if llm_cfg.get("enabled", False):
            done = self._time("llm_report_generation")
            llm_feedback = build_llm_feedback(report.to_dict(), llm_cfg)
            if llm_feedback is not None:
                report.llm_feedback = llm_feedback
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
        help="Comma-separated slide boundaries in seconds, e.g. 0,30,60,90.",
    )
    parser.add_argument(
        "--alignment-mode",
        default=None,
        choices=["manual", "auto", "hybrid"],
        help="Alignment strategy. Omit to use config or manual/auto fallback.",
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
    parser.add_argument("--vlm-enabled", action="store_true", help="Force-enable VLM slide captioning for alignment.")
    parser.add_argument(
        "--vlm-high-quality",
        action="store_true",
        help="Force-enable VLM captioning with high image detail. This may increase latency/cost.",
    )
    parser.add_argument("--vlm-model", default=None, help="Override VLM caption model name")
    parser.add_argument("--vlm-detail", default=None, choices=["low", "high", "auto"], help="Override VLM image detail")
    parser.add_argument("--vlm-cache-dir", default=None, help="Override VLM caption cache directory")
    parser.add_argument("--vlm-timeout-sec", default=None, type=float, help="Override VLM request timeout")
    parser.add_argument("--vlm-dpi", default=None, type=int, help="Override PDF render DPI for VLM images")
    parser.add_argument("--vlm-no-cache", action="store_true", help="Disable VLM caption cache for this run")
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
            },
            "vlm_caption": {
                "enabled": True if (args.vlm_enabled or args.vlm_high_quality) else None,
                "model": args.vlm_model,
                "detail": args.vlm_detail or ("high" if args.vlm_high_quality else None),
                "cache_dir": args.vlm_cache_dir,
                "timeout_sec": args.vlm_timeout_sec,
                "dpi": args.vlm_dpi,
                "cache_enabled": False if args.vlm_no_cache else None,
            },
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
        alignment_mode=args.alignment_mode,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
