import tarfile
from pathlib import Path

import pytest
import torch

from speechpt.attitude.ae_probe_runtime import ARTIFACT_URI_MARKER, AEProbePrediction, resolve_probe_path


def test_resolve_probe_path_extracts_from_local_tar(tmp_path: Path):
    source = tmp_path / "source.pt"
    torch.save({"fc1.weight": torch.zeros((256, 1024))}, source)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(source, arcname="ae_probe.pt")

    resolved = resolve_probe_path({"model_dir": str(model_dir)})

    assert resolved == model_dir / "ae_probe.pt"
    assert resolved.exists()


def test_resolve_probe_path_extracts_lora_adapter_from_tar(tmp_path: Path):
    source = tmp_path / "source.pt"
    torch.save({"fc1.weight": torch.zeros((256, 1024))}, source)
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text('{"peft_type":"LORA"}', encoding="utf-8")
    meta = tmp_path / "meta.pt"
    torch.save({"use_lora": True}, meta)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(source, arcname="ae_probe.pt")
        tf.add(adapter_config, arcname="lora_adapter/adapter_config.json")
        tf.add(meta, arcname="meta.pt")

    resolved = resolve_probe_path({"model_dir": str(model_dir)})

    assert resolved == model_dir / "ae_probe.pt"
    assert (model_dir / "lora_adapter" / "adapter_config.json").exists()
    assert torch.load(model_dir / "meta.pt", map_location="cpu")["use_lora"] is True


def test_resolve_probe_path_redownloads_when_artifact_uri_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stale_dir = tmp_path / "model"
    stale_dir.mkdir()
    torch.save({"fc1.weight": torch.zeros((256, 1024))}, stale_dir / "ae_probe.pt")
    (stale_dir / ARTIFACT_URI_MARKER).write_text("s3://bucket/old/model.tar.gz", encoding="utf-8")

    new_source = tmp_path / "new.pt"
    torch.save({"fc1.weight": torch.ones((256, 1024))}, new_source)
    new_adapter = tmp_path / "new_adapter_config.json"
    new_adapter.write_text('{"peft_type":"LORA"}', encoding="utf-8")
    new_tar = tmp_path / "new_model.tar.gz"
    with tarfile.open(new_tar, "w:gz") as tf:
        tf.add(new_source, arcname="ae_probe.pt")
        tf.add(new_adapter, arcname="lora_adapter/adapter_config.json")

    class FakeS3Client:
        def download_file(self, bucket: str, key: str, dest: str) -> None:
            Path(dest).write_bytes(new_tar.read_bytes())

    class FakeSession:
        def __init__(self, profile_name: str | None = None):
            self.profile_name = profile_name

        def client(self, name: str):
            assert name == "s3"
            return FakeS3Client()

    import boto3

    monkeypatch.setattr(boto3, "Session", FakeSession)

    resolved = resolve_probe_path(
        {
            "model_dir": str(stale_dir),
            "model_artifact_s3": "s3://bucket/new/model.tar.gz",
        }
    )

    state = torch.load(resolved, map_location="cpu")
    assert torch.all(state["fc1.weight"] == 1)
    assert (stale_dir / "lora_adapter" / "adapter_config.json").exists()
    assert (stale_dir / ARTIFACT_URI_MARKER).read_text(encoding="utf-8") == "s3://bucket/new/model.tar.gz"


def test_ae_probe_prediction_uses_report_feature_prefix():
    prediction = AEProbePrediction(
        slide_id=1,
        start_sec=0.0,
        end_sec=5.0,
        speech_rate=1.2,
        silence_ratio=0.3,
        energy_drop=0,
        energy_drop_prob=0.1,
        pitch_shift=1,
        pitch_shift_prob=0.8,
        overall_delivery=0.7,
    )

    features = prediction.to_feature_dict()

    assert features["ae_probe_speech_rate"] == 1.2
    assert features["ae_probe_pitch_shift"] == 1.0
    assert all(key.startswith("ae_probe_") for key in features)
