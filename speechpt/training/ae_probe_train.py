"""AE lightweight probe training on frozen wav2vec2 embeddings.

Manifest JSONL format:
{"audio_path":"...","speech_rate":3.1,"silence_ratio":0.2,"energy_drop":0,"pitch_shift":1,"overall_delivery":0.74}
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import io

import boto3
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2Processor


@dataclass
class Sample:
    audio_path: str
    speech_rate: float
    silence_ratio: float
    energy_drop: float
    pitch_shift: float
    overall_delivery: float


class AudioDataset(Dataset):
    def __init__(
        self,
        rows: List[Sample],
        sample_rate: int,
        chunk_sec: float,
        base_dir: Path | None = None,
        audio_dir: Path | None = None,
        audio_index: Dict[str, Path] | None = None,
        audio_s3_uri: str = "",
    ):
        self.rows = rows
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_sec)
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self._basename_cache: Dict[str, Path] = {}
        self._audio_index = audio_index or {}
        self._warn_count = 0
        # S3 직접 스트리밍용 — FastFile 한글 경로 FUSE 버그 우회
        self._audio_s3_uri = audio_s3_uri.rstrip("/") + "/" if audio_s3_uri else ""
        if self._audio_s3_uri.startswith("s3://"):
            body = self._audio_s3_uri[5:]
            self._s3_bucket, _, prefix = body.partition("/")
            self._s3_prefix = prefix
        else:
            self._s3_bucket = ""
            self._s3_prefix = ""
        self._s3_client = boto3.client("s3") if self._s3_bucket else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_path = resolve_audio_path(
            row.audio_path,
            base_dir=self.base_dir,
            audio_dir=self.audio_dir,
            basename_cache=self._basename_cache,
            audio_index=self._audio_index,
        )
        try:
            wav, _ = librosa.load(str(audio_path), sr=self.sample_rate)
        except Exception as local_exc:
            # FastFile + 한글 S3 prefix 조합에서 FUSE read가 실패하는 경우
            # boto3로 S3에서 직접 스트리밍해 우회한다.
            wav = None
            if self._s3_client is not None:
                s3_key = self._s3_prefix + Path(audio_path).name
                try:
                    obj = self._s3_client.get_object(Bucket=self._s3_bucket, Key=s3_key)
                    audio_bytes = io.BytesIO(obj["Body"].read())
                    wav, _ = librosa.load(audio_bytes, sr=self.sample_rate)
                except Exception as s3_exc:
                    if self._warn_count < 30:
                        print(json.dumps({
                            "warning": "audio_load_failed",
                            "path": str(audio_path),
                            "s3_key": s3_key,
                            "local_error": str(local_exc),
                            "s3_error": str(s3_exc),
                        }))
                    self._warn_count += 1
                    return None
            else:
                if self._warn_count < 30:
                    print(json.dumps({"warning": "audio_load_failed", "path": str(audio_path), "error": str(local_exc)}))
                self._warn_count += 1
                return None
        if len(wav) >= self.chunk_len:
            wav = wav[: self.chunk_len]
        else:
            pad = self.chunk_len - len(wav)
            wav = torch.nn.functional.pad(torch.tensor(wav), (0, pad)).numpy()
        target = torch.tensor(
            [row.speech_rate, row.silence_ratio, row.energy_drop, row.pitch_shift, row.overall_delivery],
            dtype=torch.float32,
        )
        return wav, target


def collate_skip_none(batch):
    """Collate function that drops samples where __getitem__ returned None.

    Returns None when the entire batch is empty so callers can skip it.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    wavs = torch.stack([torch.from_numpy(b[0]) for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return wavs, targets


class AEProbe(nn.Module):
    def __init__(self, in_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def read_jsonl(path: Path) -> List[Sample]:
    rows: List[Sample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(Sample(**obj))
    return rows


def split_rows(rows: List[Sample], train_ratio: float, seed: int) -> tuple[List[Sample], List[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--split-ratio must be in (0, 1)")
    if len(rows) < 2:
        raise ValueError("At least 2 rows are required to split train/valid")

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_ratio))
    n_train = min(n_train, len(shuffled) - 1)
    return shuffled[:n_train], shuffled[n_train:]


def resolve_data_paths(args) -> tuple[Path, Path]:
    if args.train and args.valid:
        return Path(args.train), Path(args.valid)

    input_dir = Path(args.input_dir)
    if input_dir.is_dir():
        train_candidates = ["train.jsonl", "ae_train.jsonl"]
        valid_candidates = ["valid.jsonl", "ae_valid.jsonl"]
        for train_name in train_candidates:
            for valid_name in valid_candidates:
                train_path = input_dir / train_name
                valid_path = input_dir / valid_name
                if train_path.exists() and valid_path.exists():
                    return train_path, valid_path

        single_candidates = ["dataset.jsonl", "manifest.jsonl", "data.jsonl", "ae.jsonl", "train.jsonl"]
        for name in single_candidates:
            path = input_dir / name
            if path.exists():
                rows = read_jsonl(path)
                train_rows, valid_rows = split_rows(rows, train_ratio=args.split_ratio, seed=args.seed)
                split_dir = Path(args.output) / "_auto_split"
                split_dir.mkdir(parents=True, exist_ok=True)
                train_path = split_dir / "train.jsonl"
                valid_path = split_dir / "valid.jsonl"
                with train_path.open("w", encoding="utf-8") as fp:
                    for row in train_rows:
                        fp.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
                with valid_path.open("w", encoding="utf-8") as fp:
                    for row in valid_rows:
                        fp.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "message": "auto split created",
                            "source": str(path),
                            "train_rows": len(train_rows),
                            "valid_rows": len(valid_rows),
                        },
                        ensure_ascii=False,
                    )
                )
                return train_path, valid_path

    raise ValueError(
        "Could not resolve train/valid data. Provide --train and --valid, or put "
        "train.jsonl+valid.jsonl (or a single manifest.jsonl/data.jsonl) under --input-dir."
    )


def build_s3_audio_index(audio_s3_uri: str, audio_dir: Path) -> Dict[str, Path]:
    """FastFile 모드에서 디렉토리 리스팅이 불가능할 때 S3 오브젝트 목록으로 인덱스 구축.

    S3 오브젝트 키에서 상대 경로를 추출하고 FastFile 마운트 경로와 조합해
    {basename: local_path} 매핑을 반환한다.
    """
    if not audio_s3_uri.startswith("s3://"):
        return {}
    body = audio_s3_uri[5:]
    bucket, _, prefix = body.partition("/")
    prefix = prefix.rstrip("/") + "/"

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    index: Dict[str, Path] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.lower().endswith(".wav"):
                continue
            rel_path = key[len(prefix):]
            local_path = audio_dir / rel_path
            index.setdefault(Path(rel_path).name, local_path)

    print(json.dumps({"s3_audio_index_built": len(index), "audio_s3_uri": audio_s3_uri}, ensure_ascii=False))
    return index


def resolve_audio_path(
    raw_path: str,
    base_dir: Path | None,
    audio_dir: Path | None,
    basename_cache: Dict[str, Path] | None = None,
    audio_index: Dict[str, Path] | None = None,
) -> Path:
    audio_path = Path(raw_path)
    if audio_path.is_absolute():
        return audio_path

    if audio_dir is not None:
        text = raw_path.strip()
        if text.startswith("audio/"):
            text = text[len("audio/") :]
        candidate = audio_dir / text
        if candidate.exists():
            return candidate

        # Fallback: some manifests contain only basename while files live under nested dirs.
        base_name = Path(text).name
        if audio_index is not None and base_name in audio_index:
            return audio_index[base_name]
        if basename_cache is not None and base_name in basename_cache:
            return basename_cache[base_name]
        matches = list(audio_dir.rglob(base_name))
        if matches:
            if basename_cache is not None:
                basename_cache[base_name] = matches[0]
            return matches[0]
        return candidate

    if base_dir is not None:
        return base_dir / audio_path
    return audio_path


def train_epoch(loader, processor, backbone, probe, optimizer, device):
    probe.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        if batch is None:
            continue
        wavs, targets = batch
        targets = targets.to(device)
        inputs = processor(list(wavs.numpy()), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hidden = backbone(**inputs).last_hidden_state
            pooled = hidden.mean(dim=1)
        preds = probe(pooled)

        reg_pred = preds[:, [0, 1, 4]]
        reg_true = targets[:, [0, 1, 4]]
        cls_pred = preds[:, [2, 3]]
        cls_true = targets[:, [2, 3]]

        loss = F.mse_loss(reg_pred, reg_true) + F.binary_cross_entropy_with_logits(cls_pred, cls_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def eval_epoch(loader, processor, backbone, probe, device):
    probe.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            wavs, targets = batch
            targets = targets.to(device)
            inputs = processor(list(wavs.numpy()), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
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


def load_training_state(probe: nn.Module, optimizer: torch.optim.Optimizer, path: Path, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        probe.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_loss = float(state.get("best_loss", float("inf")))
    else:
        probe.load_state_dict(state)
        start_epoch = 1
        best_loss = float("inf")
    return start_epoch, best_loss


def main():
    parser = argparse.ArgumentParser(description="Train AE probe on frozen wav2vec2")
    parser.add_argument("--train")
    parser.add_argument("--valid")
    parser.add_argument("--input-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output", default=os.environ.get("SM_MODEL_DIR", "artifacts/ae_model"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("SM_CHECKPOINT_DIR"))
    parser.add_argument("--resume-from")
    parser.add_argument("--audio-dir", default=os.environ.get("SM_CHANNEL_AUDIO"))
    parser.add_argument("--audio-s3", default=os.environ.get("AE_AUDIO_S3", ""))
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path, valid_path = resolve_data_paths(args)
    print(json.dumps({"train_path": str(train_path), "valid_path": str(valid_path)}, ensure_ascii=False))
    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    base_dir = train_path.parent
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    audio_index: Dict[str, Path] = {}
    if audio_dir is not None and audio_dir.exists():
        for p in audio_dir.rglob("*.wav"):
            audio_index.setdefault(p.name, p)
    print(
        json.dumps(
            {
                "audio_dir": str(audio_dir) if audio_dir else None,
                "audio_indexed_files": len(audio_index),
            },
            ensure_ascii=False,
        )
    )
    # FastFile 모드는 디렉토리 리스팅을 지원하지 않아 rglob이 0개를 반환한다.
    # 이 경우 S3 오브젝트 목록을 직접 조회해 FastFile 접근 경로를 역산한다.
    if len(audio_index) == 0 and args.audio_s3 and audio_dir is not None:
        print(json.dumps({"message": "audio_index_empty_building_from_s3", "audio_s3": args.audio_s3}, ensure_ascii=False))
        audio_index = build_s3_audio_index(args.audio_s3, audio_dir)
    train_ds = AudioDataset(
        train_rows,
        sample_rate=args.sample_rate,
        chunk_sec=args.chunk_sec,
        base_dir=base_dir,
        audio_dir=audio_dir,
        audio_index=audio_index,
        audio_s3_uri=args.audio_s3,
    )
    valid_ds = AudioDataset(
        valid_rows,
        sample_rate=args.sample_rate,
        chunk_sec=args.chunk_sec,
        base_dir=base_dir,
        audio_dir=audio_dir,
        audio_index=audio_index,
        audio_s3_uri=args.audio_s3,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_skip_none,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_skip_none,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    backbone = Wav2Vec2Model.from_pretrained(args.model).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    in_dim = int(getattr(backbone.config, "hidden_size", 768))
    print(json.dumps({"probe_input_dim": in_dim}, ensure_ascii=False))
    probe = AEProbe(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    start_epoch = 1
    best_loss = float("inf")
    resume_path = None
    if args.resume_from:
        cand = Path(args.resume_from)
        if cand.exists():
            resume_path = cand
    if resume_path is None and args.checkpoint_dir:
        cand = Path(args.checkpoint_dir) / "ae_probe_latest.pt"
        if cand.exists():
            resume_path = cand
    if resume_path is not None:
        start_epoch, best_loss = load_training_state(probe, optimizer, resume_path, device)
        print(
            json.dumps(
                {
                    "message": "resumed",
                    "resume_path": str(resume_path),
                    "start_epoch": start_epoch,
                    "best_loss": best_loss,
                },
                ensure_ascii=False,
            )
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(train_loader, processor, backbone, probe, optimizer, device)
        valid_loss = eval_epoch(valid_loader, processor, backbone, probe, device)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}, ensure_ascii=False))

        latest_state = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict": probe.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if checkpoint_dir is not None:
            torch.save(latest_state, checkpoint_dir / "ae_probe_latest.pt")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(probe.state_dict(), Path(args.output) / "ae_probe.pt")
            if checkpoint_dir is not None:
                best_state = {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(best_state, checkpoint_dir / "ae_probe_best.pt")

    torch.save(
        {"model": args.model, "sample_rate": args.sample_rate, "chunk_sec": args.chunk_sec, "dropout": 0.1},
        Path(args.output) / "meta.pt",
    )


if __name__ == "__main__":
    main()
