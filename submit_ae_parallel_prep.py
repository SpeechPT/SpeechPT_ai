"""병렬 AE 전처리 제출 스크립트.

3개의 SageMaker Training Job을 동시에 launch:
  - 잡 A: Training data 전반부 [0:half)
  - 잡 B: Training data 후반부 [half:end)
  - 잡 C: Validation data 전체 (fresh 프로세스 → ProcessPool 안정)

모두 같은 S3 prefix에 결과를 쓴다:
  s3://{bucket}/pipeline/ae/parallel-{ts}/processed/
    ├─ rows_a.jsonl          (잡 A 출력)
    ├─ rows_b.jsonl          (잡 B 출력)
    └─ eval_validation.jsonl (잡 C 출력)

병렬 잡들이 완료되면 `merge_ae_parts.py`를 돌려 train/valid/test.jsonl을
생성한다. 그 후 pipeline_ae.py를 Step 2(학습)부터 재개하면 됨.

사용법:
    python submit_ae_parallel_prep.py
    python submit_ae_parallel_prep.py --total-files 8000 --half 4000  # 테스트
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


ROLE = "arn:aws:iam::242071452299:role/SpeechPT-SageMaker-Role"
IMAGE_URI = "242071452299.dkr.ecr.ap-northeast-2.amazonaws.com/speechpt-training:latest"
BUCKET = "aws-s3-speechpt1"
REGION = "ap-northeast-2"

LABELS_S3 = f"s3://{BUCKET}/datasets/raws/Training/02.라벨링데이터/"
AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Training/01.원천데이터/"
VAL_LABELS_S3 = f"s3://{BUCKET}/datasets/raws/Validation/02.라벨링데이터/"
VAL_AUDIO_S3 = f"s3://{BUCKET}/datasets/raws/Validation/01.원천데이터/"


def make_estimator(name: str, hp: dict, output_s3: str, instance_type: str, max_run: int):
    return Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type=instance_type,
        volume_size=50,
        input_mode="File",
        max_run=max_run,
        output_path=output_s3,
        entry_point="train.py",
        source_dir="./speechpt/training/",
        hyperparameters=hp,
        sagemaker_session=sagemaker.Session(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-files", type=int, default=0,
                        help="0=전체. 테스트용 작은 값 가능.")
    parser.add_argument("--half", type=int, default=34000,
                        help="잡 A의 end_index (잡 B는 이 값부터 시작). "
                             "--total-files > 0일 때 자동 조정.")
    parser.add_argument("--prep-instance-type", default="ml.g5.2xlarge")
    parser.add_argument("--val-instance-type", default="ml.g5.xlarge",
                        help="Validation 잡 인스턴스. 쿼터 분배 (Training 2개가 g5.2xl 다 씀).")
    parser.add_argument("--max-run-sec", type=int, default=86400,
                        help="잡당 최대 실행 시간 (초). 기본 24h.")
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Validation 잡 launch 안 함 (이미 있는 경우 등).")
    parser.add_argument("--training-only-job", choices=["a", "b", "both"], default="both",
                        help="둘 다 / 하나만 launch (디버깅).")
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_prefix = f"s3://{BUCKET}/pipeline/ae/parallel-{ts}/processed/"
    job_output_path = f"s3://{BUCKET}/pipeline/ae/parallel-{ts}/job-output/"

    # 분할: total-files 지정 시 절반씩, 아니면 전체에서 half 기준 분할
    if args.total_files > 0:
        half = args.total_files // 2
        end_a = half
        start_b = half
        end_b = args.total_files
    else:
        end_a = args.half
        start_b = args.half
        end_b = 0  # 0 = 끝까지

    print(f"output_prefix: {output_prefix}")
    print(f"split: A=[0:{end_a}), B=[{start_b}:{end_b if end_b else 'end'})")

    common_hp_a = {
        "labels-s3-uri": LABELS_S3,
        "audio-s3-uri": AUDIO_S3,
        "output-s3-uri": output_prefix,
        "use-audio-features": "true",
        "s3-rescue-on-empty": "true",
        "checkpoint-interval": args.checkpoint_interval,
        "skip-validation-prep": "true",  # 병렬 잡은 Validation 안 함
        "start-index": 0,
        "end-index": end_a,
        "output-suffix": "_a",
    }
    common_hp_b = {
        **common_hp_a,
        "start-index": start_b,
        "end-index": end_b,
        "output-suffix": "_b",
    }

    estimator_a = make_estimator(
        name=f"prep-a-{ts}",
        hp=common_hp_a,
        output_s3=job_output_path,
        instance_type=args.prep_instance_type,
        max_run=args.max_run_sec,
    )
    estimator_b = make_estimator(
        name=f"prep-b-{ts}",
        hp=common_hp_b,
        output_s3=job_output_path,
        instance_type=args.prep_instance_type,
        max_run=args.max_run_sec,
    )

    inputs = {"labels": TrainingInput(s3_data=LABELS_S3, input_mode="File")}

    submitted = []
    if args.training_only_job in ("a", "both"):
        job_a_name = f"speechpt-prep-a-{ts}"
        estimator_a.fit(inputs=inputs, job_name=job_a_name, wait=False)
        submitted.append(("training-a", job_a_name))
    if args.training_only_job in ("b", "both"):
        job_b_name = f"speechpt-prep-b-{ts}"
        estimator_b.fit(inputs=inputs, job_name=job_b_name, wait=False)
        submitted.append(("training-b", job_b_name))

    # Validation 잡은 별도 (fresh 프로세스 → ProcessPool 안정)
    if not args.skip_validation:
        common_hp_val = {
            # labels-s3-uri는 SageMaker 채널용으로 필요하지만 validation_only면 안 씀
            "labels-s3-uri": VAL_LABELS_S3,
            "audio-s3-uri": VAL_AUDIO_S3,
            "output-s3-uri": output_prefix,
            "use-audio-features": "true",
            "s3-rescue-on-empty": "true",
            "checkpoint-interval": args.checkpoint_interval,
            "validation-only": "true",
            "val-labels-s3-uri": VAL_LABELS_S3,
            "val-audio-s3-uri": VAL_AUDIO_S3,
        }
        estimator_val = make_estimator(
            name=f"prep-val-{ts}",
            hp=common_hp_val,
            output_s3=job_output_path,
            instance_type=args.val_instance_type,
            max_run=args.max_run_sec,
        )
        job_val_name = f"speechpt-prep-val-{ts}"
        estimator_val.fit(
            inputs={"labels": TrainingInput(s3_data=VAL_LABELS_S3, input_mode="File")},
            job_name=job_val_name,
            wait=False,
        )
        submitted.append(("validation", job_val_name))

    print()
    print(f"=== {len(submitted)} 잡 제출 완료 ===")
    for role, name in submitted:
        print(f"  [{role}] {name}")
    print()
    print(f"S3 출력 prefix: {output_prefix}")
    print()
    print("다음 단계 (모든 잡 완료 후):")
    print(f"  python merge_ae_parts.py --output-s3 {output_prefix}")
    print()
    print("진행 상황 확인:")
    print(f"  aws sagemaker list-training-jobs --region {REGION} --status-equals InProgress")
    print()


if __name__ == "__main__":
    main()
