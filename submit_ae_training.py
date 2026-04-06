import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch
from datetime import datetime
import os

# === 설정 ===
role = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
bucket = "aws-s3-speechpt1"
input_s3 = os.environ.get("AE_INPUT_S3", f"s3://{bucket}/datasets/processed/ae/")
audio_s3 = os.environ.get("AE_AUDIO_S3", f"s3://{bucket}/datasets/raws/Training/01.원천데이터/")
model_name = os.environ.get("AE_MODEL", "facebook/wav2vec2-base")
job_name = os.environ.get("AE_JOB_NAME", f"speechpt-ae-train-v1-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
instance_type = os.environ.get("AE_INSTANCE_TYPE", "ml.g5.xlarge")
instance_count = int(os.environ.get("AE_INSTANCE_COUNT", "1"))
input_mode = os.environ.get("AE_INPUT_MODE", "FastFile")
resume_from = os.environ.get("AE_RESUME_FROM")
batch_size = int(os.environ.get("AE_BATCH_SIZE", "8"))
epochs = int(os.environ.get("AE_EPOCHS", "10"))
lr = float(os.environ.get("AE_LR", "1e-3"))
chunk_sec = int(os.environ.get("AE_CHUNK_SEC", "30"))

session = sagemaker.Session()
use_spot = False

estimator_kwargs = {
    "framework_version": "2.1",
    "py_version": "py310",
    "role": role,
    "instance_count": instance_count,
    "instance_type": instance_type,
    "input_mode": input_mode,
    "use_spot_instances": use_spot,
    "max_run": 86400,
    "output_path": f"s3://{bucket}/models/ae/v1/",
    "checkpoint_s3_uri": f"s3://{bucket}/checkpoints/ae/v1/",
    "hyperparameters": {
        "batch-size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "model": model_name,
        "chunk-sec": chunk_sec,
        "audio-s3": audio_s3,
    },
    "entry_point": "train.py",
    "source_dir": "./speechpt/training/",
    "sagemaker_session": session,
}
if resume_from:
    estimator_kwargs["hyperparameters"]["resume-from"] = resume_from
if use_spot:
    estimator_kwargs["max_wait"] = 86400

estimator = PyTorch(**estimator_kwargs)

estimator.fit(
    inputs={
        "training": TrainingInput(
            s3_data=input_s3,
            input_mode=input_mode,
        ),
        "audio": TrainingInput(
            s3_data=audio_s3,
            input_mode=input_mode,
        ),
    },
    job_name=job_name,
    wait=False,
)

print("Training Job 제출 완료!")
print(f"job_name={job_name}")
print(f"instance_type={instance_type}, instance_count={instance_count}")
print(f"input_mode={input_mode}")
print(f"batch_size={batch_size}, epochs={epochs}, lr={lr}, chunk_sec={chunk_sec}")
print(f"training_s3={input_s3}")
print(f"audio_s3={audio_s3}")
print("SageMaker 콘솔 또는 Studio에서 진행 상황을 확인하세요.")
