import json
from pathlib import Path

from src.reporting import build_dashboard


def test_dashboard_is_derived_from_recorded_artifacts(tmp_path: Path) -> None:
    metadata = {
        "history": [
            {
                "train/box_loss": 1.2,
                "metrics/mAP50(B)": 0.2,
                "metrics/mAP50-95(B)": 0.1,
            },
            {
                "train/box_loss": 0.9,
                "metrics/mAP50(B)": 0.3,
                "metrics/mAP50-95(B)": 0.2,
            },
        ]
    }
    metrics = {
        "metrics": {
            "metrics/precision(B)": 0.4,
            "metrics/recall(B)": 0.5,
            "metrics/mAP50(B)": 0.3,
            "metrics/mAP50-95(B)": 0.2,
        }
    }
    metadata_path = tmp_path / "metadata.json"
    metrics_path = tmp_path / "metrics.json"
    output = tmp_path / "dashboard.png"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    build_dashboard(metadata_path, metrics_path, output)
    assert output.is_file() and output.stat().st_size > 0
