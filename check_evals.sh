EVAL_JOBS=(
    "speechpt-ae-eval-v1-20260408-014509"
    "speechpt-ae-eval-v1-20260406-073124"
    "speechpt-ae-eval-v1-20260331-kor"
    "speechpt-ae-eval-v1-20260331-orig"
    "speechpt-ae-eval-v1-20260326-100158"
)

# 배열 순회
for EVAL_JOB in "${EVAL_JOBS[@]}"; do
    echo "=== ${EVAL_JOB} ==="
    
    # python3를 사용하여 파이썬 코드 실행
    python3 -c "$(cat <<PYTHON_EOF
import boto3, tarfile, json, os
from pathlib import Path

job_name = "${EVAL_JOB}"
s3_bucket = "aws-s3-speechpt1"
s3_key = f"models/ae-eval/v1/{job_name}/output/model.tar.gz"

s3 = boto3.client('s3', region_name='ap-northeast-2')

out = Path(f'/tmp/ae_eval_result/{job_name}')
out.mkdir(parents=True, exist_ok=True)
tar_path = out / 'model.tar.gz'

try:
    s3.download_file(s3_bucket, s3_key, str(tar_path))
    
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(out)
        
    result_file = out / 'eval_result.json'
    if result_file.exists():
        result = json.loads(result_file.read_text())
        print(json.dumps(result, indent=2))
    else:
        print(f"  결과 없음: {result_file} 파일이 존재하지 않습니다.")
        
except Exception as e:
    print(f"  오류 발생: {e}")
PYTHON_EOF
    )"
    echo ""
done
