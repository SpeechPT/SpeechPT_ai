import json
from pathlib import Path


def test_examples_have_required_top_level_keys():
    base = Path("examples")

    example_input = json.loads((base / "example_input.json").read_text())
    assert "document_path" in example_input
    assert "audio_path" in example_input
    assert "slide_change_times" in example_input

    ce_output = json.loads((base / "example_ce_output.json").read_text())
    assert isinstance(ce_output, list)
    assert "slide_id" in ce_output[0]
    assert "coverage" in ce_output[0]

    ae_output = json.loads((base / "example_ae_output.json").read_text())
    assert isinstance(ae_output, list)
    assert "slide_id" in ae_output[0]
    assert "features" in ae_output[0]

    report = json.loads((base / "example_report.json").read_text())
    assert "overall_scores" in report
    assert "highlight_sections" in report
