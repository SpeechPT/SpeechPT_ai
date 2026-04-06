from datetime import datetime
import os

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


role = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
image_uri = "242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest"
bucket = "aws-s3-speechpt1"

labels_s3 = os.environ.get("AE_LABELS_S3", f"s3://{bucket}/datasets/raws/Training/02.라벨링데이터/")
audio_s3 = os.environ.get("AE_AUDIO_S3", f"s3://{bucket}/datasets/raws/Training/01.원천데이터/")
processed_s3 = os.environ.get("AE_PROCESSED_S3", f"s3://{bucket}/datasets/processed/ae/full/")
target_rows = int(os.environ.get("AE_PREP_TARGET_ROWS", "0"))
require_readable = os.environ.get("AE_PREP_REQUIRE_READABLE_AUDIO", "true")
use_audio_features = os.environ.get("AE_PREP_USE_AUDIO_FEATURES", "true")
allow_label_only_fallback = os.environ.get("AE_PREP_ALLOW_LABEL_ONLY_FALLBACK", "false")
s3_rescue_on_empty = os.environ.get("AE_PREP_S3_RESCUE_ON_EMPTY", "true")
max_files = int(os.environ.get("AE_PREP_MAX_FILES", "0"))

instance_type = os.environ.get("AE_PREP_INSTANCE_TYPE", "ml.m5.xlarge")
instance_count = int(os.environ.get("AE_PREP_INSTANCE_COUNT", "1"))
volume_size_gb = int(os.environ.get("AE_PREP_VOLUME_SIZE_GB", "50"))
input_mode = os.environ.get("AE_PREP_INPUT_MODE", "File")
include_audio = os.environ.get("AE_PREP_INCLUDE_AUDIO", "false").lower() in {"1", "true", "yes", "y"}
job_name = os.environ.get("AE_PREP_JOB_NAME", f"speechpt-ae-prep-v1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
wait = os.environ.get("AE_PREP_WAIT", "true").lower() in {"1", "true", "yes", "y"}

session = sagemaker.Session()

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    volume_size=volume_size_gb,
    input_mode=input_mode,
    max_run=86400,
    output_path=f"s3://{bucket}/models/ae-prep/",
    entry_point="train.py",
    source_dir="./speechpt/training/",
    hyperparameters={
        "output-s3-uri": processed_s3,
        "labels-s3-uri": labels_s3,
        "audio-s3-uri": audio_s3,
        "target-rows": target_rows,
        "require-readable-audio": require_readable,
        "use-audio-features": use_audio_features,
        "allow-label-only-fallback": allow_label_only_fallback,
        "s3-rescue-on-empty": s3_rescue_on_empty,
        "max-files": max_files,
    },
    sagemaker_session=session,
)

inputs = {
    "labels": TrainingInput(
        s3_data=labels_s3,
        input_mode=input_mode,
    ),
}
if include_audio:
    inputs["audio"] = TrainingInput(
        s3_data=audio_s3,
        input_mode=input_mode,
    )

estimator.fit(inputs=inputs, job_name=job_name, wait=wait)

print("Preprocessing Job 제출 완료!")
print(f"job_name={job_name}")
print(f"instance_type={instance_type}, instance_count={instance_count}")
print(f"input_mode={input_mode}")
print(f"labels_s3={labels_s3}")
print(f"audio_s3={audio_s3}")
print(f"include_audio={include_audio}")
print(f"processed_s3={processed_s3}")
print(f"target_rows={target_rows}")
print(f"require_readable_audio={require_readable}")
print(f"use_audio_features={use_audio_features}")
print(f"allow_label_only_fallback={allow_label_only_fallback}")
print(f"s3_rescue_on_empty={s3_rescue_on_empty}")
print(f"max_files={max_files}")
print(f"wait={wait}")
print(f"volume_size_gb={volume_size_gb}")
