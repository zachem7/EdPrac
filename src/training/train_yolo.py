from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils.reproducibility import get_device
from src.utils.save_metrics import save_experiment


def get_yolo_device(config: dict[str, Any]) -> int | str:
    device = get_device(config.get("device", "auto"))
    return 0 if device.type == "cuda" else "cpu"


def train_yolo(model_name: str, config: dict[str, Any]) -> dict[str, float]:
    model_cfg = config["models"][model_name]
    train_cfg = config["training"]
    data_yaml = Path(config["data"]["processed_dir"]) / "gtsdb_yolo.yaml"
    save_dir = Path(train_cfg["save_dir"]).resolve()

    model = YOLO(model_cfg["weights"])
    results = model.train(
        data=str(data_yaml),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(model_cfg.get("image_size", config["data"]["image_size"])),
        batch=int(train_cfg["batch_size"]),
        lr0=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optimizer=str(train_cfg.get("optimizer", "AdamW")),
        device=get_yolo_device(config),
        project=str(save_dir),
        name=model_name,
        exist_ok=True,
    )

    val_metrics = model.val(
        data=str(data_yaml),
        split="test",
        device=get_yolo_device(config),
        project=str(save_dir),
        name=f"{model_name}_val",
        exist_ok=True,
    )
    metrics = {
        "map": float(val_metrics.box.map),
        "map50": float(val_metrics.box.map50),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "f1": 2
        * float(val_metrics.box.mp)
        * float(val_metrics.box.mr)
        / max(float(val_metrics.box.mp) + float(val_metrics.box.mr), 1e-8),
        "train_results_dir": str(results.save_dir),
    }

    save_experiment(
        model_name=model_name,
        hyperparameters={**train_cfg, **model_cfg},
        metrics=metrics,
        csv_path=config["results"]["metrics_csv"],
        jsonl_path=config["results"]["metrics_jsonl"],
    )
    return metrics
