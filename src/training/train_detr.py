from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.gtsdb_dataset import GTSDBDetectionDataset, detection_collate
from src.evaluation.evaluate_detr import evaluate_detr
from src.evaluation.evaluate_detr import get_detr_eval_settings
from src.models.detr import build_detr
from src.training.train_torchvision import create_optimizer
from src.utils.reproducibility import get_device
from src.utils.save_metrics import save_experiment


def xyxy_to_cxcywh_normalized(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    cx = ((x1 + x2) / 2) / image_size
    cy = ((y1 + y2) / 2) / image_size
    width = (x2 - x1) / image_size
    height = (y2 - y1) / image_size
    return torch.stack([cx, cy, width, height], dim=1).clamp(0, 1)


def build_detr_labels(targets: list[dict], image_size: int, device) -> list[dict]:
    labels = []
    for target in targets:
        labels.append(
            {
                "class_labels": (target["labels"].to(device) - 1).long(),
                "boxes": xyxy_to_cxcywh_normalized(target["boxes"].to(device), image_size),
            }
        )
    return labels


def train_detr(model_name: str, config: dict[str, Any]) -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    processed_dir = Path(config["data"]["processed_dir"])
    image_size = int(config["data"]["image_size"])
    train_cfg = config["training"]
    model_cfg = config["models"][model_name]

    train_dataset = GTSDBDetectionDataset(processed_dir / "manifest_train.csv", image_size=image_size, train=True)
    val_dataset = GTSDBDetectionDataset(processed_dir / "manifest_val.csv", image_size=image_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=detection_collate,
    )

    model = build_detr(model_name, config).to(device)
    optimizer = create_optimizer(model, config, model_cfg)
    score_threshold, top_k = get_detr_eval_settings(config, model_name)
    grad_clip = float(model_cfg.get("grad_clip", 0.1))
    save_dir = Path(train_cfg["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.csv"
    history_path.write_text("epoch,train_loss,map,map50,precision,recall,f1\n", encoding="utf-8")

    best_map = -1.0
    best_score = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float] = {}
    total_epochs = int(train_cfg["epochs"])
    for epoch in range(1, total_epochs + 1):
        print(f"\n=== {model_name}: epoch {epoch}/{total_epochs} ===")
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"train {epoch}/{total_epochs}", leave=False):
            pixel_values = torch.stack([image.to(device) for image in images])
            labels = build_detr_labels(targets, image_size, device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "DETR loss became NaN/Inf. Lower learning_rate or restart training."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate_detr(
            model,
            val_loader,
            device,
            image_size,
            score_threshold,
            top_k,
            desc=f"eval {epoch}/{total_epochs}",
        )
        history_path.write_text(
            history_path.read_text(encoding="utf-8")
            + f"{epoch},{train_loss:.6f},{metrics['map']:.6f},{metrics['map50']:.6f},"
            + f"{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f}\n",
            encoding="utf-8",
        )

        score = (metrics["map50"], metrics["map"], metrics["f1"])
        if score >= best_score:
            best_score = score
            best_map = metrics["map"]
            best_metrics = metrics
            torch.save(model.state_dict(), save_dir / "best.pth")

    best_metrics["checkpoint"] = str(save_dir / "best.pth")
    best_metrics["history"] = str(history_path)
    save_experiment(
        model_name=model_name,
        hyperparameters={**train_cfg, **config["models"][model_name]},
        metrics=best_metrics,
        csv_path=config["results"]["metrics_csv"],
        jsonl_path=config["results"]["metrics_jsonl"],
    )
    return best_metrics
