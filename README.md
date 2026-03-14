# SpeechPT Engine

SpeechPT는 발표 자료와 음성을 함께 분석해 코칭 리포트를 생성한다.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests
```bash
source .venv/bin/activate
python -m pytest tests
```

## Run Report from Example Outputs
```bash
python -m speechpt.report.report_generator \
  --ce examples/example_ce_output.json \
  --ae examples/example_ae_output.json \
  --tpl speechpt/report/templates/feedback_ko.yaml
```

## Run End-to-End Pipeline (manual whisper words)
```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /abs/path/slides.pdf \
  --audio /abs/path/audio.wav \
  --slide-timestamps 0,30,60,90 \
  --whisper-json /abs/path/whisper_words.json
```

## Run End-to-End Pipeline (auto STT)
1. `configs/pipeline_v0.1.yaml`에서 `stt.enabled: true`로 설정
2. 다음 명령 실행
```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /abs/path/slides.pdf \
  --audio /abs/path/audio.wav \
  --slide-timestamps 0,30,60,90
```

## Train CE LoRA
```bash
python -m speechpt.training.ce_lora_train \
  --model klue/roberta-base \
  --train /abs/path/ce_train.jsonl \
  --valid /abs/path/ce_valid.jsonl \
  --output artifacts/ce_model
```

## Train AE Probe (frozen wav2vec2)
```bash
python -m speechpt.training.ae_probe_train \
  --train /abs/path/ae_train.jsonl \
  --valid /abs/path/ae_valid.jsonl \
  --output artifacts/ae_model
```

## Evaluation
```bash
python eval/eval_coherence.py \
  --gold eval/data/coherence_gold.json \
  --pred examples/example_ce_output.json

python eval/eval_attitude.py \
  --gold eval/data/attitude_gold.json \
  --pred examples/example_ae_output.json

python eval/ablation.py
```

## Project Layout
- `speechpt/coherence`: CE 파이프라인 모듈
- `speechpt/attitude`: AE 파이프라인 모듈
- `speechpt/stt`: Whisper STT 연동
- `speechpt/report`: 통합 리포트 생성
- `speechpt/training`: CE/AE 학습 스크립트
- `speechpt/pipeline.py`: 전체 오케스트레이션
- `eval`: 평가 및 어블레이션 스크립트
- `examples`: 예시 입출력 JSON
- `Model_Development_Plan.md`: MLOps 전달용 개발 계획
- `Training_Only_Plan.md`: 학습 전용 계획
