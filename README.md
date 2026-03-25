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

## SageMaker: AE preprocess in cloud (no local full download)
```bash
python3 submit_ae_preprocessing.py
```

기본 입력/출력:
- labels: `s3://aws-s3-speechpt1/datasets/raws/Training/02.라벨링데이터/`
- audio: `s3://aws-s3-speechpt1/datasets/raws/Training/01.원천데이터/`
- processed output: `s3://aws-s3-speechpt1/datasets/processed/ae/full/`

전처리 결과는 `all.jsonl` + `train/valid/test.jsonl`을 생성합니다.

## AE Best Practice (recommended)
1. 전처리 전체는 **한 번만** 실행해서 `full/all.jsonl` 생성
2. 학습은 `all.jsonl`에서 부분집합을 뽑아 반복

부분집합 생성 예시:
```bash
python3 -m speechpt.training.make_ae_subset \
  --input /abs/path/all.jsonl \
  --output-dir /abs/path/ae_subset_10k \
  --max-rows 10000
```

그 뒤 subset `train/valid`만 S3에 올리고 학습:
```bash
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/exp-10k/ \
AE_INSTANCE_TYPE=ml.g5.xlarge \
python3 submit_ae_training.py
```

추가학습(resume) 예시:
```bash
AE_INPUT_S3=s3://aws-s3-speechpt1/datasets/processed/ae/exp-30k/ \
AE_INSTANCE_TYPE=ml.g5.xlarge \
python3 submit_ae_training.py
```

`submit_ae_training.py`는 동일 `checkpoint_s3_uri`를 사용하면 자동으로 최신 체크포인트를 복원합니다.
필요하면 `AE_RESUME_FROM=/opt/ml/checkpoints/ae_probe_best.pt`로 명시할 수 있습니다.

## SageMaker: AE end-to-end (preprocess -> train)
```bash
python3 submit_ae_end_to_end.py
```

## Prepare AE dataset (csv/json/jsonl -> train/valid/test jsonl)
```bash
python -m speechpt.training.prepare_ae_dataset \
  --input /abs/path/ae_raw.csv \
  --output-dir /abs/path/ae_prepared \
  --audio-root /abs/path/audio
```

출력 파일:
- `/abs/path/ae_prepared/train.jsonl`
- `/abs/path/ae_prepared/valid.jsonl`
- `/abs/path/ae_prepared/test.jsonl`

SageMaker 채널(`SM_CHANNEL_TRAINING`)에 `train.jsonl`, `valid.jsonl`을 넣으면
`ae_probe_train.py`는 `--train/--valid` 없이도 자동으로 파일을 찾습니다.
단일 파일(`manifest.jsonl`/`data.jsonl`)만 있으면 자동 분할해서 학습합니다.

## Prepare AE dataset from SpeechPT raws (label json + wav)
```bash
python -m speechpt.training.prepare_ae_from_raws \
  --label-dir /abs/path/02.라벨링데이터 \
  --audio-dir /abs/path/01.원천데이터 \
  --output-dir /abs/path/ae_prepared
```

주의:
- 이 스크립트는 raw 메타 + 음성 신호 기반으로 학습 타깃을 휴리스틱 생성합니다.
- 정답 점수가 별도로 있으면 그 값을 우선 사용하세요.

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
