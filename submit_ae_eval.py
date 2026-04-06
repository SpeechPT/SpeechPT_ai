from datetime import datetime
import os

import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


role = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
bucket = "aws-s3-speechpt1"
input_s3 = os.environ.get("AE_INPUT_S3", f"s3://{bucket}/datasets/processed/ae/audio-v2/")
audio_s3 = os.environ.get("AE_AUDIO_S3", f"s3://{bucket}/datasets/raws/Training/01.원천데이터/")
model_artifact_s3 = os.environ.get(
    "AE_MODEL_ARTIFACT_S3",
    f"s3://{bucket}/models/ae/v1/speechpt-ae-train-v1-20260403-104158/output/model.tar.gz",
)
backbone_model = os.environ.get("AE_MODEL", "kresnik/wav2vec2-large-xlsr-korean")
job_name = os.environ.get("AE_EVAL_JOB_NAME", f"speechpt-ae-eval-v1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
instance_type = os.environ.get("AE_EVAL_INSTANCE_TYPE", "ml.g5.xlarge")
input_mode = os.environ.get("AE_EVAL_INPUT_MODE", "FastFile")
max_test_samples = int(os.environ.get("AE_EVAL_MAX_TEST_SAMPLES", "0"))
wait = os.environ.get("AE_EVAL_WAIT", "true").lower() in {"1", "true", "yes", "y"}

session = sagemaker.Session()

estimator = PyTorch(
    framework_version="2.1",
    py_version="py310",
    role=role,
    instance_count=1,
    instance_type=instance_type,
    input_mode=input_mode,
    max_run=86400,
    output_path=f"s3://{bucket}/models/ae-eval/v1/",
    entry_point="ae_probe_eval.py",
    source_dir="./speechpt/training/",
    hyperparameters={
        "model-artifact-s3-uri": model_artifact_s3,
        "model": backbone_model,
        "chunk-sec": 20,
        "batch-size": 2,
        "max-test-samples": max_test_samples,
        "audio-s3": audio_s3,
    },
    sagemaker_session=session,
)

estimator.fit(
    inputs={
        "training": TrainingInput(s3_data=input_s3, input_mode=input_mode),
        "audio": TrainingInput(s3_data=audio_s3, input_mode=input_mode),
    },
    job_name=job_name,
    wait=wait,
)

print("AE eval Job 제출 완료!")
print(f"job_name={job_name}")
print(f"instance_type={instance_type}")
print(f"input_s3={input_s3}")
print(f"audio_s3={audio_s3}")
print(f"model_artifact_s3={model_artifact_s3}")
print(f"backbone_model={backbone_model}")
print(f"wait={wait}")
