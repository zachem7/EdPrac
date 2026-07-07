from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.training.train_yolo import get_yolo_device


def evaluate_yolo(
    weights_path: str | Path,
    config: dict[str, Any],
    split: str = "test",
) -> dict[str, float]:
    data_yaml = Path(config["data"]["processed_dir"]) / "gtsdb_yolo.yaml"
    model = YOLO(str(weights_path))
    result = model.val(data=str(data_yaml), split=split, device=get_yolo_device(config))
    precision = float(result.box.mp)
    recall = float(result.box.mr)
    return {
        "map": float(result.box.map),
        "map50": float(result.box.map50),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }
