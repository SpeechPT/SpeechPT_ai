import tarfile
from pathlib import Path

import torch

from speechpt.attitude.ae_probe_runtime import AEProbePrediction, resolve_probe_path


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
