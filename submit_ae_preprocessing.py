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

instance_type = os.environ.get("AE_PREP_INSTANCE_TYPE", "ml.m5.xlarge")
instance_count = int(os.environ.get("AE_PREP_INSTANCE_COUNT", "1"))
input_mode = os.environ.get("AE_PREP_INPUT_MODE", "FastFile")
job_name = os.environ.get("AE_PREP_JOB_NAME", f"speechpt-ae-prep-v1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
wait = os.environ.get("AE_PREP_WAIT", "true").lower() in {"1", "true", "yes", "y"}

session = sagemaker.Session()

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    input_mode=input_mode,
    max_run=86400,
    output_path=f"s3://{bucket}/models/ae-prep/",
    entry_point="train.py",
    source_dir="./speechpt/training/",
    hyperparameters={
        "output-s3-uri": processed_s3,
        "labels-s3-uri": labels_s3,
    },
    sagemaker_session=session,
)

estimator.fit(
    inputs={
        "labels": TrainingInput(
            s3_data=labels_s3,
            input_mode=input_mode,
        ),
        "audio": TrainingInput(
            s3_data=audio_s3,
            input_mode=input_mode,
        ),
    },
    job_name=job_name,
    wait=wait,
)

print("Preprocessing Job 제출 완료!")
print(f"job_name={job_name}")
print(f"instance_type={instance_type}, instance_count={instance_count}")
print(f"input_mode={input_mode}")
print(f"labels_s3={labels_s3}")
print(f"audio_s3={audio_s3}")
print(f"processed_s3={processed_s3}")
print(f"wait={wait}")
