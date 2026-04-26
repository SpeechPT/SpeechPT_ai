"""SpeechPT AE — SageMaker Pipeline (전처리 → 학습 → 평가).

사용법:
    # 소규모 테스트 (라벨 100개만)
    python pipeline_ae.py --max-files 100

    # 전체 데이터
    python pipeline_ae.py

    # 파라미터 커스텀
    python pipeline_ae.py --epochs 15 --lr 1e-4 --batch-size 16
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


# === AWS 설정 ===
ROLE = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
BUCKET = "aws-s3-speechpt1"
REGION = "ap-northeast-2"
PIPELINE_NAME = "SpeechPT-AE-Pipeline"
IMAGE_URI = "242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest"

# S3 경로
LABELS_S3 = f"s3://{BUCKET}/datasets/raws/Training/02.라벨링데이터/"
AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Training/01.원천데이터/"

# 프레임워크
FRAMEWORK_VERSION = "2.1"
PY_VERSION = "py310"
SOURCE_DIR = "./speechpt/training/"


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

    processed_s3 = f"s3://{BUCKET}/pipeline/ae/{ts}/processed/"

    # ===== Step 1: 전처리 (TrainingStep) =====
    # Processing Job에 ml.c5 할당량이 없으므로 기존과 동일하게 Estimator로 실행.
    # 전처리 결과를 S3에 업로드하고, 학습 Step이 그 경로를 참조한다.
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
    # 학습된 모델 artifact를 받아서 test set으로 평가.
    # ml.t3.xlarge 사용 (processing job 할당량 내).
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
            "--audio-s3", AUDIO_S3,
            "--chunk-sec", "20",
            "--batch-size", "2",
        ],
    )

    step_eval = ProcessingStep(
        name="AE-Evaluate",
        step_args=step_eval_args,
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
        ],
        steps=[step_preprocess, step_train, step_eval],
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
        "--prep-instance-type", default="ml.m5.xlarge", help="전처리 인스턴스 (Training Job)"
    )
    parser.add_argument(
        "--train-instance-type", default="ml.g5.xlarge", help="학습 인스턴스"
    )
    parser.add_argument(
        "--eval-instance-type", default="ml.t3.xlarge", help="평가 인스턴스 (Processing Job)"
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
    print(f"  파라미터: MaxFiles={args.max_files}, Epochs={args.epochs}, LR={args.lr}, BatchSize={args.batch_size}")
    print()
    print("진행 상황 확인:")
    print(f"  SageMaker 콘솔 → Pipelines → {PIPELINE_NAME}")
    print()
    print("CLI로 상태 확인:")
    print(f"  aws sagemaker list-pipeline-executions --pipeline-name {PIPELINE_NAME} --region {REGION}")


if __name__ == "__main__":
    main()
