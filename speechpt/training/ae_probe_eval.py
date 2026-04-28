from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import boto3
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from ae_probe_train import AEProbe, AudioDataset, collate_skip_none, read_jsonl


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    body = uri[5:]
    bucket, _, key = body.partition("/")
    return bucket, key


def download_and_extract_model(model_s3_uri: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket, key = parse_s3_uri(model_s3_uri)
    tar_path = out_dir / "model.tar.gz"
    boto3.client("s3").download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    model_path = out_dir / "ae_probe.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"ae_probe.pt not found in {model_s3_uri}")
    return model_path


def eval_loss(loader, processor, backbone, probe, device, sample_rate: int):
    probe.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            wavs, targets = batch
            targets = targets.to(device)
            inputs = processor(list(wavs.numpy()), sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
            hidden = backbone(**inputs).last_hidden_state
            pooled = hidden.mean(dim=1)
            preds = probe(pooled)
            reg_pred = preds[:, [0, 1, 4]]
            reg_true = targets[:, [0, 1, 4]]
            cls_pred = preds[:, [2, 3]]
            cls_true = targets[:, [2, 3]]
            loss = F.mse_loss(reg_pred, reg_true) + F.binary_cross_entropy_with_logits(cls_pred, cls_true)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser(description="Compare AE finetuned vs base(random probe) on test set")
    parser.add_argument("--input-dir", default="/opt/ml/input/data/training")
    parser.add_argument("--audio-dir", default="/opt/ml/input/data/audio")
    parser.add_argument("--model", default="kresnik/wav2vec2-large-xlsr-korean")
    parser.add_argument("--model-artifact-s3-uri", default="")
    parser.add_argument("--model-local-dir", default="", help="Local dir with model.tar.gz (pipeline mode). Skips S3 download.")
    parser.add_argument("--audio-s3", default="", help="S3 URI for audio fallback (e.g. s3://bucket/prefix/)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="", help="Output dir for eval_result.json (pipeline mode).")
    parser.add_argument("--eval-file", default="", help="JSONL file name for evaluation (e.g. eval_validation.jsonl). Falls back to test.jsonl.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)
    audio_dir = Path(args.audio_dir)
    # eval_validation.jsonl (AIHub Validation 세트) 우선, 없으면 test.jsonl 폴백
    if args.eval_file:
        test_path = input_dir / args.eval_file
    else:
        test_path = input_dir / "eval_validation.jsonl"
        if not test_path.exists():
            test_path = input_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path.name} not found under {input_dir}")
    print(json.dumps({"eval_file": test_path.name}, ensure_ascii=False))
    rows = read_jsonl(test_path)
    if args.max_test_samples > 0:
        rows = rows[: args.max_test_samples]

    audio_index = {}
    if audio_dir.exists():
        for p in audio_dir.rglob("*.wav"):
            audio_index.setdefault(p.name, p)

    ds = AudioDataset(
        rows,
        sample_rate=args.sample_rate,
        chunk_sec=args.chunk_sec,
        base_dir=test_path.parent,
        audio_dir=audio_dir,
        audio_index=audio_index,
        audio_s3_uri=args.audio_s3,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_skip_none)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    backbone = Wav2Vec2Model.from_pretrained(args.model).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    in_dim = int(getattr(backbone.config, "hidden_size", 768))
    torch.manual_seed(args.seed)
    base_probe = AEProbe(in_dim=in_dim).to(device)
    base_loss = eval_loss(loader, processor, backbone, base_probe, device, sample_rate=args.sample_rate)

    # Pipeline mode: load from local dir; Standalone mode: download from S3
    if args.model_local_dir:
        local_dir = Path(args.model_local_dir)
        tar_path = None
        for f in local_dir.iterdir():
            if f.name.endswith(".tar.gz"):
                tar_path = f
                break
        if tar_path is not None:
            extract_dir = Path("/tmp/ae_eval_model")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(extract_dir)
            model_path = extract_dir / "ae_probe.pt"
            model_base_dir = extract_dir
        else:
            model_path = local_dir / "ae_probe.pt"
            model_base_dir = local_dir
        if not model_path.exists():
            raise FileNotFoundError(f"ae_probe.pt not found in {local_dir}")
    else:
        model_path = download_and_extract_model(args.model_artifact_s3_uri, Path("/tmp/ae_eval_model"))
        model_base_dir = model_path.parent

    # LoRA adapter 감지 및 적용
    lora_dir = model_base_dir / "lora_adapter"
    use_lora = lora_dir.exists() and (lora_dir / "adapter_config.json").exists()
    if use_lora:
        from peft import PeftModel
        finetuned_backbone = Wav2Vec2Model.from_pretrained(args.model).to(device)
        finetuned_backbone = PeftModel.from_pretrained(finetuned_backbone, str(lora_dir))
        finetuned_backbone = finetuned_backbone.merge_and_unload()
        finetuned_backbone.eval()
        for p in finetuned_backbone.parameters():
            p.requires_grad = False
        print(json.dumps({"lora_adapter_loaded": True, "lora_dir": str(lora_dir)}, ensure_ascii=False))
    else:
        finetuned_backbone = backbone

    finetuned_probe = AEProbe(in_dim=in_dim).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        finetuned_probe.load_state_dict(state["model_state_dict"])
    else:
        finetuned_probe.load_state_dict(state)
    finetuned_loss = eval_loss(loader, processor, finetuned_backbone, finetuned_probe, device, sample_rate=args.sample_rate)

    improvement_abs = base_loss - finetuned_loss
    improvement_pct = (improvement_abs / base_loss * 100.0) if base_loss > 0 else 0.0
    result = {
        "num_test_rows": len(rows),
        "audio_indexed_files": len(audio_index),
        "backbone_model": args.model,
        "probe_input_dim": in_dim,
        "use_lora": use_lora,
        "base_loss": base_loss,
        "finetuned_loss": finetuned_loss,
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
    }
    print(json.dumps(result, ensure_ascii=False))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("/opt/ml/model")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "eval_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
