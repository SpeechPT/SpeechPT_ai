"""SpeechPT AE 파이프라인 — Step 1(전처리) 스킵 버전.

이미 별도로 처리된 JSONL이 S3에 있을 때 사용. Step 2(학습) → Step 3(평가)
→ Step 4(품질 게이트) → Step 5(모델 등록)을 실행.

사용법:
    python pipeline_ae_postprep.py \\
        --processed-s3 s3://aws-s3-speechpt1/pipeline/ae/parallel-20260511-133744/processed/ \\
        --use-lora --lr 1e-4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

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


ROLE = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
BUCKET = "aws-s3-speechpt1"
REGION = "ap-northeast-2"
PIPELINE_NAME = "SpeechPT-AE-Pipeline-PostPrep"
IMAGE_URI = "242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest"

AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Training/01.원천데이터/"
VAL_AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Validation/01.원천데이터/"

FRAMEWORK_VERSION = "2.1"
PY_VERSION = "py310"
SOURCE_DIR = "./speechpt/training/"
MODEL_PACKAGE_GROUP = "SpeechPT-AE-Models"


def create_pipeline(args) -> Pipeline:
    pipeline_session = PipelineSession()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

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

    processed_s3 = args.processed_s3.rstrip("/") + "/"

    # ===== Step 2: 학습 (TrainingStep) =====
    train_estimator = PyTorch(
        entry_point="train.py",
        source_dir=SOURCE_DIR,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        role=ROLE,
        instance_count=1,
        instance_type=args.train_instance_type,
        input_mode="FastFile",
        max_run=172800,
        output_path=f"s3://{BUCKET}/pipeline/ae/postprep-{ts}/models/",
        checkpoint_s3_uri=f"s3://{BUCKET}/pipeline/ae/postprep-{ts}/checkpoints/",
        hyperparameters={
            "epochs": param_epochs,
            "lr": param_lr,
            "batch-size": param_batch_size,
            "model": param_backbone,
            "chunk-sec": 15,  # 30 → 15: GPU OOM 회피 (wav2vec2-large activations)
            "audio-s3": AUDIO_S3,
            "use-lora": param_use_lora,
            "lora-r": param_lora_r,
        },
        sagemaker_session=pipeline_session,
    )

    step_train_args = train_estimator.fit(
        inputs={
            "training": TrainingInput(s3_data=processed_s3, input_mode="FastFile"),
            "audio": TrainingInput(s3_data=AUDIO_S3, input_mode="FastFile"),
        },
    )

    step_train = TrainingStep(name="AE-Train", step_args=step_train_args)

    # ===== Step 3: 평가 (ProcessingStep on CPU - quota workaround) =====
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
                destination=f"s3://{BUCKET}/pipeline/ae/postprep-{ts}/evaluation/",
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

    # ===== Step 5: 모델 등록 =====
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

    # ===== Step 4: 품질 게이트 =====
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

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            param_epochs, param_lr, param_batch_size,
            param_backbone, param_use_lora, param_lora_r,
            param_quality_threshold, param_approval_status,
        ],
        steps=[step_train, step_eval, step_quality_gate],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-s3", required=True,
                        help="이미 처리된 train/valid/test/eval_validation.jsonl이 있는 S3 prefix.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--eval-instance-type", default="ml.t3.xlarge")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--quality-threshold", type=float, default=50.0)
    parser.add_argument("--approval-status", default="PendingManualApproval",
                        choices=["PendingManualApproval", "Approved"])
    parser.add_argument("--upsert-only", action="store_true")
    args = parser.parse_args()

    pipeline = create_pipeline(args)
    print("파이프라인 등록 중...")
    pipeline.upsert(role_arn=ROLE)
    print(f"파이프라인 등록 완료: {PIPELINE_NAME}")

    if args.upsert_only:
        return

    execution = pipeline.start(
        execution_display_name=f"ae-postprep-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
    )
    print("파이프라인 실행 시작!")
    print(f"  Execution ARN: {execution.arn}")
    print(f"  Processed S3: {args.processed_s3}")
    print(f"  LoRA r={args.lora_r}, lr={args.lr}, epochs={args.epochs}")


if __name__ == "__main__":
    main()
