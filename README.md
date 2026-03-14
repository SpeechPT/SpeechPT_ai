# SpeechPT AI Engine

SpeechPT AI Engine은 발표 문서(PPT/PDF)와 음성을 함께 분석해 코칭 리포트를 생성합니다.

## What it does
- CE(Coherence Engine): 슬라이드 핵심 내용과 발화 정합성 분석
- AE(Attitude Engine): 속도, 침묵, 에너지, 변화점 기반 전달 품질 분석
- Report Engine: 슬라이드별 이슈, 전체 점수, 피드백 리포트 생성

## Output scores
- `content_coverage`
- `delivery_stability`
- `pacing_score`

## Project structure
- `speechpt/coherence`: 문서 파싱, 키포인트 추출, 전사 정렬, CE 스코어링
- `speechpt/attitude`: 오디오 피처, 변화점, AE 스코어링, wav2vec2 임베딩
- `speechpt/stt`: Whisper 기반 STT 연동
- `speechpt/report`: 통합 리포트 생성
- `speechpt/training`: CE LoRA / AE probe 학습 스크립트
- `eval`: 평가 및 어블레이션
- `configs`: 파이프라인/학습 설정

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests
```bash
python -m pytest tests
```

## Run pipeline (manual STT JSON)
```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /abs/path/slides.pdf \
  --audio /abs/path/audio.wav \
  --slide-timestamps 0,30,60,90 \
  --whisper-json /abs/path/whisper_words.json
```

`whisper_words.json` format:
```json
{
  "words": [
    {"word": "안녕하세요", "start": 0.0, "end": 0.4},
    {"word": "발표", "start": 0.5, "end": 0.8}
  ]
}
```

## Run pipeline (auto STT)
1. `configs/pipeline_v0.1.yaml`에서 `stt.enabled: true`로 변경
2. STT 백엔드 옵션 확인 (`backend`, `model_name`, `language`)
3. 아래 명령 실행

```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /abs/path/slides.pdf \
  --audio /abs/path/audio.wav \
  --slide-timestamps 0,30,60,90
```

## Visual analysis (image/chart/table)
현재 지원 범위:
- 시각 요소 메타 추출 (`visual_items`)
- PDF 기반 OCR (easyocr) 옵션
- 시각 캡션(`visual_captions`)을 CE 매칭에 포함
- 리포트 이슈: `visual_not_explained`

설정 예시 (`configs/pipeline_v0.1.yaml`):
```yaml
coherence:
  visual:
    enabled: true
    ocr_engine: easyocr
    ocr_languages: [ko, en]
    min_confidence: 0.3
    max_text_len: 80
```

## Train CE (LoRA)
```bash
python -m speechpt.training.ce_lora_train \
  --model klue/roberta-base \
  --train /abs/path/ce_train.jsonl \
  --valid /abs/path/ce_valid.jsonl \
  --output artifacts/ce_model
```

## Train AE (frozen wav2vec2 + probe)
```bash
python -m speechpt.training.ae_probe_train \
  --train /abs/path/ae_train.jsonl \
  --valid /abs/path/ae_valid.jsonl \
  --output artifacts/ae_model
```

## Evaluate
```bash
python eval/eval_coherence.py \
  --gold eval/data/coherence_gold.json \
  --pred examples/example_ce_output.json

python eval/eval_attitude.py \
  --gold eval/data/attitude_gold.json \
  --pred examples/example_ae_output.json

python eval/ablation.py
```

## Current status
- Engine core implemented (CE + AE + report)
- STT auto/manual path implemented
- wav2vec2 inference path integrated (`use_probe`)
- Visual OCR/caption matching integrated (PDF-first)
- Tests passing

## Notes
- STT 자동 모드는 `stt.enabled: true`가 필요합니다.
- OCR 사용 시 `easyocr` 설치가 필요합니다.
- PPT 렌더링 기반 OCR 고도화는 다음 단계입니다.
