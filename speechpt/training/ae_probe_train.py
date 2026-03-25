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
    ):
        self.rows = rows
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_sec)
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        audio_path = resolve_audio_path(
            row.audio_path,
            base_dir=self.base_dir,
            audio_dir=self.audio_dir,
        )
        wav, _ = librosa.load(audio_path, sr=self.sample_rate)
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


class AEProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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


def resolve_audio_path(raw_path: str, base_dir: Path | None, audio_dir: Path | None) -> Path:
    audio_path = Path(raw_path)
    if audio_path.is_absolute():
        return audio_path

    if audio_dir is not None:
        text = raw_path.strip()
        if text.startswith("audio/"):
            return audio_dir / text[len("audio/") :]
        return audio_dir / text

    if base_dir is not None:
        return base_dir / audio_path
    return audio_path


def train_epoch(loader, processor, backbone, probe, optimizer, device):
    probe.train()
    total_loss = 0.0
    for wavs, targets in loader:
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
    return total_loss / max(1, len(loader))


def eval_epoch(loader, processor, backbone, probe, device):
    probe.eval()
    total_loss = 0.0
    with torch.no_grad():
        for wavs, targets in loader:
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
    return total_loss / max(1, len(loader))


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
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path, valid_path = resolve_data_paths(args)
    print(json.dumps({"train_path": str(train_path), "valid_path": str(valid_path)}, ensure_ascii=False))
    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    base_dir = train_path.parent
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    print(json.dumps({"audio_dir": str(audio_dir) if audio_dir else None}, ensure_ascii=False))
    train_ds = AudioDataset(
        train_rows,
        sample_rate=args.sample_rate,
        chunk_sec=args.chunk_sec,
        base_dir=base_dir,
        audio_dir=audio_dir,
    )
    valid_ds = AudioDataset(
        valid_rows,
        sample_rate=args.sample_rate,
        chunk_sec=args.chunk_sec,
        base_dir=base_dir,
        audio_dir=audio_dir,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    backbone = Wav2Vec2Model.from_pretrained(args.model).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    probe = AEProbe().to(device)
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

    torch.save({"model": args.model, "sample_rate": args.sample_rate, "chunk_sec": args.chunk_sec}, Path(args.output) / "meta.pt")


if __name__ == "__main__":
    main()
