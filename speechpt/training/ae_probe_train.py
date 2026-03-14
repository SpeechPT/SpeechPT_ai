"""AE lightweight probe training on frozen wav2vec2 embeddings.

Manifest JSONL format:
{"audio_path":"...","speech_rate":3.1,"silence_ratio":0.2,"energy_drop":0,"pitch_shift":1,"overall_delivery":0.74}
"""
from __future__ import annotations

import argparse
import json
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
    def __init__(self, rows: List[Sample], sample_rate: int, chunk_sec: float):
        self.rows = rows
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_sec)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, _ = librosa.load(row.audio_path, sr=self.sample_rate)
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


def main():
    parser = argparse.ArgumentParser(description="Train AE probe on frozen wav2vec2")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", default="artifacts/ae_model")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = read_jsonl(Path(args.train))
    valid_rows = read_jsonl(Path(args.valid))
    train_ds = AudioDataset(train_rows, sample_rate=args.sample_rate, chunk_sec=args.chunk_sec)
    valid_ds = AudioDataset(valid_rows, sample_rate=args.sample_rate, chunk_sec=args.chunk_sec)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    backbone = Wav2Vec2Model.from_pretrained(args.model).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    probe = AEProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    best_loss = float("inf")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(train_loader, processor, backbone, probe, optimizer, device)
        valid_loss = eval_epoch(valid_loader, processor, backbone, probe, device)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}, ensure_ascii=False))
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(probe.state_dict(), Path(args.output) / "ae_probe.pt")

    torch.save({"model": args.model, "sample_rate": args.sample_rate, "chunk_sec": args.chunk_sec}, Path(args.output) / "meta.pt")


if __name__ == "__main__":
    main()
