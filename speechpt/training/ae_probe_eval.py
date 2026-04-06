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
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--model-artifact-s3-uri", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)
    audio_dir = Path(args.audio_dir)
    test_path = input_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"test.jsonl not found under {input_dir}")
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

    model_path = download_and_extract_model(args.model_artifact_s3_uri, Path("/tmp/ae_eval_model"))
    finetuned_probe = AEProbe(in_dim=in_dim).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        finetuned_probe.load_state_dict(state["model_state_dict"])
    else:
        finetuned_probe.load_state_dict(state)
    finetuned_loss = eval_loss(loader, processor, backbone, finetuned_probe, device, sample_rate=args.sample_rate)

    improvement_abs = base_loss - finetuned_loss
    improvement_pct = (improvement_abs / base_loss * 100.0) if base_loss > 0 else 0.0
    result = {
        "num_test_rows": len(rows),
        "audio_indexed_files": len(audio_index),
        "backbone_model": args.model,
        "probe_input_dim": in_dim,
        "base_loss": base_loss,
        "finetuned_loss": finetuned_loss,
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
    }
    print(json.dumps(result, ensure_ascii=False))

    out = Path("/opt/ml/model/eval_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
