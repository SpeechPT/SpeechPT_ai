# SpeechPT AI Engine

SpeechPT AI Engine은 발표 자료(PDF/PPT)와 발표 음성을 함께 분석해 슬라이드 단위 코칭 리포트를 생성합니다.

핵심 파이프라인:

```text
PDF/PPT + audio
→ STT(word timestamp)
→ CE slide alignment / coherence
→ AE speech attitude
→ deterministic report JSON
→ optional LLM feedback JSON
```

---

## 현재 런타임 기준

### CE: 내용 정합성 / 자동정렬

- 슬라이드 텍스트, OCR, VLM caption 신호를 자동정렬에 사용할 수 있습니다.
- 기본 config에서는 VLM이 꺼져 있습니다. 필요 시 CLI override로 켭니다.
- CE coverage는 슬라이드 충실도 채점이 아니라, 슬라이드 구간과 발화의 정합성 신호입니다.
- 표지/목차/감사 슬라이드는 내용 평균 점수에서 제외될 수 있습니다.

### AE: 말하기 태도 / 전달 안정성

AE는 발표 내용이 맞는지가 아니라 말하는 방식을 봅니다.

현재 AE는 다음 구조입니다.

```text
실측 feature 중심 점수
+ LoRA AE probe 보조 신호
```

사용 중인 AE artifact:

```text
s3://aws-s3-speechpt1/pipeline/ae/postprep-20260512-023656/models/pipelines-3qm07layk5io-AE-Train-BoZ9elGx9Q/output/model.tar.gz
```

artifact 구성:

```text
ae_probe.pt
lora_adapter/
meta.pt
```

런타임 로딩 구조:

```text
kresnik/wav2vec2-large-xlsr-korean
+ lora_adapter merge
+ ae_probe.pt head
```

주의:

- `ae_probe_*` 값은 raw model output입니다.
- 최종 `delivery_stability`는 raw `ae_probe_overall_delivery`를 그대로 쓰지 않습니다.
- 현재 최종 점수는 실측 `silence_ratio`, `words_per_sec`, `filler_count`, `dwell_z` 중심이고, probe는 기본 15%만 반영합니다.
- `ae_probe_energy_drop`, `ae_probe_pitch_shift`는 JSON에는 보존하지만 사용자 피드백 직접 근거로 쓰지 않습니다.

기본 AE 점수 가중치:

```text
silence_ratio 실측        35%
speech_rate/words_per_sec 25%
filler density            15%
dwell balance             10%
LoRA overall_delivery     15%
```

---

## 프로젝트 구조

```text
speechpt/
├── coherence/       CE: 슬라이드-발화 정렬, 내용 정합성, VLM caption
├── attitude/        AE: 속도, 침묵, dwell, 간투사, AE probe runtime
├── stt/             Whisper/faster-whisper STT
├── report/          리포트 생성, LLM feedback, snapshot
└── training/        SageMaker AE 학습/평가/개발용 추론 스크립트

configs/
├── pipeline_v0.1.yaml      로컬/개발용 기본 config
└── pipeline_runtime.yaml   SageMaker endpoint 런타임 config
```

---

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

SageMaker/AWS 기능을 쓸 경우:

```bash
aws configure
aws sts get-caller-identity
```

OpenAI LLM/VLM 기능을 쓸 경우:

```bash
export OPENAI_API_KEY="..."
```

또는 config의 `report.llm.api_key_file`에 지정된 파일을 사용합니다.

---

## E2E 실행

### 1. STT JSON이 이미 있을 때

가장 안정적인 개발/검증 방식입니다.

```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.mp3 \
  --whisper-json /path/to/whisper_words.json \
  --alignment-mode hybrid \
  > report.json
```

`whisper_words.json` 형식:

```json
{
  "words": [
    {"word": "안녕하세요", "start": 0.0, "end": 0.4},
    {"word": "발표", "start": 0.5, "end": 0.8}
  ]
}
```

### 2. 로컬 CPU에서 STT까지 포함

```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.mp3 \
  --stt-enabled \
  --stt-backend faster-whisper \
  --stt-model large-v3-turbo \
  --stt-device cpu \
  --stt-compute-type int8 \
  --save-whisper-json /tmp/whisper_words.json \
  > report.json
```

CPU에서는 느릴 수 있습니다. SageMaker/GPU 환경에서는 STT 모델을 더 크게 쓰는 것이 권장됩니다.

### 3. VLM caption까지 포함

VLM은 슬라이드 이미지 분석을 CE 자동정렬 신호로만 사용합니다.

```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.mp3 \
  --whisper-json /path/to/whisper_words.json \
  --vlm-enabled \
  > report.json
```

고품질 이미지 분석:

```bash
python -m speechpt.pipeline \
  --config configs/pipeline_v0.1.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.mp3 \
  --whisper-json /path/to/whisper_words.json \
  --vlm-high-quality \
  > report.json
```

### 4. LLM 최종 피드백까지 포함

로컬 기본 config는 LLM이 꺼져 있습니다. 임시 config를 만들어 켭니다.

```bash
cp configs/pipeline_v0.1.yaml /tmp/speechpt_pipeline_llm.yaml
python3 - <<'PY'
from pathlib import Path
import yaml

p = Path("/tmp/speechpt_pipeline_llm.yaml")
cfg = yaml.safe_load(p.read_text())
cfg.setdefault("report", {}).setdefault("llm", {})["enabled"] = True
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
PY

OPENAI_API_KEY="$OPENAI_API_KEY" python -m speechpt.pipeline \
  --config /tmp/speechpt_pipeline_llm.yaml \
  --document /path/to/slides.pdf \
  --audio /path/to/audio.mp3 \
  --stt-enabled \
  --stt-backend faster-whisper \
  --stt-model large-v3-turbo \
  --stt-device cpu \
  --stt-compute-type int8 \
  > report_with_llm.json
```

SageMaker endpoint에서는 `configs/pipeline_runtime.yaml`이 LLM enabled 기준입니다.

---

## 주요 출력 JSON

E2E 결과는 stdout으로 하나의 report JSON을 출력합니다.

중요 필드:

| 필드 | 설명 |
|------|------|
| `overall_scores.content_coverage` | 표지/감사 등 boilerplate 제외한 내용 정합성 점수 |
| `overall_scores.content_coverage_all` | 모든 슬라이드 포함 CE 평균 |
| `overall_scores.delivery_stability` | AE 최종 전달 안정성 점수 |
| `overall_scores.pacing_score` | 슬라이드별 발화 속도 일관성 |
| `alignment.final_boundaries` | 자동정렬된 슬라이드 경계 |
| `alignment.confidence` | 정렬 신뢰도 |
| `transcript_segments` | 슬라이드별 STT 텍스트와 word timestamp |
| `per_slide_detail[].ae_probe` | LoRA AE probe raw output |
| `per_slide_detail[].delivery_stability` | 슬라이드별 AE composite 점수 |
| `llm_feedback` | LLM 최종 요약/개선 액션. LLM enabled일 때만 |

백엔드가 “대사 누르면 해당 슬라이드로 이동” 기능을 만들 때는 `transcript_segments`를 쓰면 됩니다.

```json
{
  "slide_id": 3,
  "start_sec": 24.6,
  "end_sec": 45.3,
  "text": "이 슬라이드에서는 ...",
  "words": [
    {"word": "이", "start": 24.6, "end": 24.7}
  ]
}
```

---

## AE 단독 추론

AE만 빠르게 확인하고 싶을 때 사용합니다. 이 명령도 E2E와 같은 runtime인 `speechpt.attitude.ae_probe_runtime`을 호출합니다.

```bash
python -m speechpt.training.ae_probe_infer \
  --audio /path/to/sample.wav \
  --model-dir /tmp/ae_model_lora \
  --model-artifact-s3 s3://aws-s3-speechpt1/pipeline/ae/postprep-20260512-023656/models/pipelines-3qm07layk5io-AE-Train-BoZ9elGx9Q/output/model.tar.gz \
  --probe-chunk-sec 8 \
  --device cpu
```

슬라이드 경계별 점수:

```bash
python -m speechpt.training.ae_probe_infer \
  --audio /path/to/sample.wav \
  --model-dir /tmp/ae_model_lora \
  --model-artifact-s3 s3://aws-s3-speechpt1/pipeline/ae/postprep-20260512-023656/models/pipelines-3qm07layk5io-AE-Train-BoZ9elGx9Q/output/model.tar.gz \
  --slide-timestamps 0,40,90,130 \
  --probe-chunk-sec 8 \
  --device cpu \
  --output-jsonl /tmp/ae_probe_output.jsonl
```

출력되는 5개 raw score:

| 필드 | 의미 |
|------|------|
| `speech_rate` | 모델이 예측한 발화 속도 |
| `silence_ratio` | 모델이 예측한 침묵 비율 |
| `energy_drop` | 에너지 저하 여부 |
| `pitch_shift` | 피치 변화 여부 |
| `overall_delivery` | 모델 raw 전달력 |

이 값은 디버깅용 raw output입니다. 최종 리포트 점수는 `report_generator.py`에서 실측 feature와 함께 다시 계산됩니다.

---

## Calibration snapshot

리포트의 점수/이슈/정렬 confidence만 뽑아 회귀 비교용 JSON을 만듭니다. 원본 오디오나 전체 transcript는 저장하지 않습니다.

```bash
python -m speechpt.report.snapshot \
  --report /path/to/report.json \
  --case controlv_large \
  --out /path/to/controlv_snapshot.json
```

이전 snapshot과 비교:

```bash
python -m speechpt.report.snapshot \
  --report /path/to/report.json \
  --case controlv_large \
  --baseline /path/to/previous_snapshot.json \
  --out /path/to/current_snapshot.json
```

---

## AE 학습 파이프라인

`pipeline_ae.py`는 SageMaker에서 AE 전처리 → 학습 → 평가 → 품질 게이트 → 모델 등록을 수행합니다.

```text
전처리            학습                  평가                 품질 게이트
ml.c5.4xlarge     ml.g5.2xlarge         ml.g5.xlarge         improvement_pct 기준
```

소규모 테스트:

```bash
python pipeline_ae.py --max-files 100
```

전체 학습:

```bash
python pipeline_ae.py
```

LoRA 학습:

```bash
python pipeline_ae.py --use-lora --lr 1e-4
```

평가 결과 확인:

```bash
aws s3 ls s3://aws-s3-speechpt1/pipeline/ae/ --region ap-northeast-2
aws s3 cp s3://aws-s3-speechpt1/pipeline/ae/<timestamp>/evaluation/eval_result.json - \
  --region ap-northeast-2
```

최근 SageMaker training job:

```bash
aws sagemaker list-training-jobs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 10 \
  --region ap-northeast-2
```

CloudWatch 로그:

```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --follow \
  --region ap-northeast-2
```

---

## Filler Word Detection

STT word timestamp에서 “어”, “음”, “그러니까” 같은 간투사를 탐지합니다.

```python
from speechpt.attitude.filler_detector import detect_fillers

words = [
    {"word": "안녕하세요", "start": 0.0, "end": 0.4},
    {"word": "어", "start": 1.2, "end": 1.4},
]

result = detect_fillers(words, slide_timestamps=[0, 40, 90])
```

메인 파이프라인에서는 `attitude_scorer.py`가 이 공용 detector를 사용합니다.
