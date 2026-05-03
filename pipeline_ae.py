"""SpeechPT AE — SageMaker Pipeline (전처리 → 학습 → 평가 → 품질 게이트 → 모델 등록).

사용법:
    # 소규모 테스트 (라벨 100개만)
    python pipeline_ae.py --max-files 100

    # 전체 데이터
    python pipeline_ae.py

    # LoRA 파인튜닝
    python pipeline_ae.py --use-lora --lr 1e-4

    # 파라미터 커스텀
    python pipeline_ae.py --epochs 15 --lr 1e-4 --batch-size 16
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


# === AWS 설정 ===
ROLE = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
BUCKET = "aws-s3-speechpt1"
REGION = "ap-northeast-2"
PIPELINE_NAME = "SpeechPT-AE-Pipeline"
IMAGE_URI = "242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest"

# S3 경로 — Training 데이터
LABELS_S3 = f"s3://{BUCKET}/datasets/raws/Training/02.라벨링데이터/"
AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Training/01.원천데이터/"

# S3 경로 — Validation 데이터 (독립 평가셋)
VAL_LABELS_S3 = f"s3://{BUCKET}/datasets/raws/Validation/02.라벨링데이터/"
VAL_AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Validation/01.원천데이터/"

# 프레임워크
FRAMEWORK_VERSION = "2.1"
PY_VERSION = "py310"
SOURCE_DIR = "./speechpt/training/"

# 모델 레지스트리
MODEL_PACKAGE_GROUP = "SpeechPT-AE-Models"


def create_pipeline(args) -> Pipeline:
    pipeline_session = PipelineSession()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # ===== 파이프라인 파라미터 =====
    param_max_files = ParameterInteger(name="MaxFiles", default_value=args.max_files)
    param_epochs = ParameterInteger(name="Epochs", default_value=args.epochs)
    param_lr = ParameterString(name="LearningRate", default_value=str(args.lr))
    param_batch_size = ParameterInteger(name="BatchSize", default_value=args.batch_size)
    param_backbone = ParameterString(
        name="BackboneModel", default_value="kresnik/wav2vec2-large-xlsr-korean"
    )
    param_use_lora = ParameterString(name="UseLora", default_value=str(args.use_lora).lower())
    param_lora_r = ParameterInteger(name="LoraR", default_value=args.lora_r)
    param_quality_threshold = ParameterFloat(
        name="QualityThreshold", default_value=args.quality_threshold
    )
    param_approval_status = ParameterString(
        name="ApprovalStatus", default_value=args.approval_status
    )

    processed_s3 = f"s3://{BUCKET}/pipeline/ae/{ts}/processed/"

    # ===== Step 1: 전처리 (TrainingStep) =====
    # Processing Job에 ml.c5 할당량이 없으므로 Estimator(커스텀 Docker 이미지)로 실행.
    # Training 데이터 → train/valid/test.jsonl
    # Validation 데이터 → eval_validation.jsonl (독립 평가셋)
    prep_estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type=args.prep_instance_type,
        volume_size=50,
        max_run=86400,
        output_path=f"s3://{BUCKET}/pipeline/ae/{ts}/prep-output/",
        entry_point="train.py",
        source_dir=SOURCE_DIR,
        hyperparameters={
            "output-s3-uri": processed_s3,
            "labels-s3-uri": LABELS_S3,
            "audio-s3-uri": AUDIO_S3,
            "use-audio-features": "true",
            "s3-rescue-on-empty": "true",
            "max-files": param_max_files,
            "val-labels-s3-uri": VAL_LABELS_S3,
            "val-audio-s3-uri": VAL_AUDIO_S3,
        },
        sagemaker_session=pipeline_session,
    )

    step_prep_args = prep_estimator.fit(
        inputs={
            "labels": TrainingInput(s3_data=LABELS_S3, input_mode="File"),
        },
    )

    step_preprocess = TrainingStep(
        name="AE-Preprocess",
        step_args=step_prep_args,
    )

    # ===== Step 2: 학습 (TrainingStep) =====
    # 전처리가 S3에 업로드한 JSONL을 읽어서 학습.
    # depends_on으로 전처리 완료를 보장.
    train_estimator = PyTorch(
        entry_point="train.py",
        source_dir=SOURCE_DIR,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
        instance_count=1,
        instance_type=args.train_instance_type,
        input_mode="FastFile",
        max_run=86400,
        output_path=f"s3://{BUCKET}/pipeline/ae/{ts}/models/",
        checkpoint_s3_uri=f"s3://{BUCKET}/pipeline/ae/{ts}/checkpoints/",
        hyperparameters={
            "epochs": param_epochs,
            "lr": param_lr,
            "batch-size": param_batch_size,
            "model": param_backbone,
            "chunk-sec": 30,
            "audio-s3": AUDIO_S3,
            "use-lora": param_use_lora,
            "lora-r": param_lora_r,
        },
        sagemaker_session=pipeline_session,
    )

    step_train_args = train_estimator.fit(
        inputs={
            "training": TrainingInput(
                s3_data=processed_s3,
                input_mode="FastFile",
            ),
            "audio": TrainingInput(
                s3_data=AUDIO_S3,
                input_mode="FastFile",
            ),
        },
    )

    step_train = TrainingStep(
        name="AE-Train",
        step_args=step_train_args,
        depends_on=[step_preprocess],
    )

    # ===== Step 3: 평가 (ProcessingStep) =====
    # 학습된 모델 artifact를 Validation 데이터(독립 평가셋)로 평가.
    # eval_validation.jsonl 사용 + Validation 오디오 S3 스트리밍.
    eval_processor = PyTorchProcessor(
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
        instance_count=1,
        instance_type=args.eval_instance_type,
        volume_size_in_gb=30,
        sagemaker_session=pipeline_session,
    )

    step_eval_args = eval_processor.run(
        code="ae_probe_eval.py",
        source_dir=SOURCE_DIR,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model/",
                input_name="model",
            ),
            ProcessingInput(
                source=processed_s3,
                destination="/opt/ml/processing/input/data/",
                input_name="data",
            ),
            ProcessingInput(
                source=VAL_AUDIO_S3,
                destination="/opt/ml/processing/input/audio/",
                input_name="audio",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/",
                destination=f"s3://{BUCKET}/pipeline/ae/{ts}/evaluation/",
            ),
        ],
        arguments=[
            "--input-dir", "/opt/ml/processing/input/data/",
            "--audio-dir", "/opt/ml/processing/input/audio/",
            "--model-local-dir", "/opt/ml/processing/input/model/",
            "--output-dir", "/opt/ml/processing/output/",
            "--model", "kresnik/wav2vec2-large-xlsr-korean",
            "--eval-file", "eval_validation.jsonl",
            "--audio-s3", VAL_AUDIO_S3,
            "--chunk-sec", "20",
            "--batch-size", "16",
        ],
    )

    # PropertyFile: eval_result.json에서 품질 지표 추출
    eval_report = PropertyFile(
        name="EvalReport",
        output_name="evaluation",
        path="eval_result.json",
    )

    step_eval = ProcessingStep(
        name="AE-Evaluate",
        step_args=step_eval_args,
        property_files=[eval_report],
    )

    # ===== Step 5: 모델 등록 (ModelStep) =====
    # 품질 게이트 통과 시 ModelPackageGroup에 모델 등록.
    model = Model(
        image_uri=IMAGE_URI,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=ROLE,
    )

    step_register = ModelStep(
        name="AE-RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status=param_approval_status,
        ),
    )

    # ===== Step 4: 품질 게이트 (ConditionStep) =====
    # improvement_pct >= QualityThreshold 이면 모델 등록, 아니면 스킵.
    condition_quality = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=eval_report,
            json_path="improvement_pct",
        ),
        right=param_quality_threshold,
    )

    step_quality_gate = ConditionStep(
        name="AE-QualityGate",
        conditions=[condition_quality],
        if_steps=[step_register],
        else_steps=[],
    )

    # ===== 파이프라인 조립 =====
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            param_max_files,
            param_epochs,
            param_lr,
            param_batch_size,
            param_backbone,
            param_use_lora,
            param_lora_r,
            param_quality_threshold,
            param_approval_status,
        ],
        steps=[step_preprocess, step_train, step_eval, step_quality_gate],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="SpeechPT AE SageMaker Pipeline")
    parser.add_argument("--max-files", type=int, default=0, help="전처리 라벨 수 제한 (0=무제한)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--prep-instance-type", default="ml.c5.4xlarge", help="전처리 인스턴스 (Training Job)"
    )
    parser.add_argument(
        "--train-instance-type", default="ml.g5.2xlarge", help="학습 인스턴스"
    )
    parser.add_argument(
        "--eval-instance-type", default="ml.g5.xlarge", help="평가 인스턴스 (Processing Job, GPU)"
    )
    parser.add_argument("--use-lora", action="store_true", help="LoRA 파인튜닝 활성화")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (8, 16, 32)")
    parser.add_argument(
        "--quality-threshold", type=float, default=50.0,
        help="모델 등록 기준 improvement_pct (%)"
    )
    parser.add_argument(
        "--approval-status", default="PendingManualApproval",
        choices=["PendingManualApproval", "Approved"],
        help="모델 등록 승인 상태"
    )
    parser.add_argument("--upsert-only", action="store_true", help="파이프라인 등록만 하고 실행 안 함")
    args = parser.parse_args()

    pipeline = create_pipeline(args)

    # 파이프라인 등록 (생성 또는 업데이트)
    print("파이프라인 등록 중...")
    pipeline.upsert(role_arn=ROLE)
    print(f"파이프라인 등록 완료: {PIPELINE_NAME}")

    if args.upsert_only:
        print("--upsert-only 모드: 실행은 건너뜁니다.")
        return

    # 파이프라인 실행
    execution = pipeline.start(
        execution_display_name=f"ae-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    )
    print("파이프라인 실행 시작!")
    print(f"  Execution ARN: {execution.arn}")
    lora_str = f", LoRA=r{args.lora_r}" if args.use_lora else ", LoRA=off"
    print(f"  파라미터: MaxFiles={args.max_files}, Epochs={args.epochs}, LR={args.lr}, BatchSize={args.batch_size}{lora_str}")
    print(f"  품질 기준: improvement_pct >= {args.quality_threshold}%")
    print(f"  승인 상태: {args.approval_status}")
    print()
    print("진행 상황 확인:")
    print(f"  SageMaker 콘솔 → Pipelines → {PIPELINE_NAME}")
    print()
    print("CLI로 상태 확인:")
    print(f"  aws sagemaker list-pipeline-executions --pipeline-name {PIPELINE_NAME} --region {REGION}")


if __name__ == "__main__":
    main()
