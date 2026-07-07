from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.gtsdb_dataset import GTSDBDetectionDataset, detection_collate
from src.evaluation.evaluate_torchvision import evaluate_torchvision
from src.models.model_factory import create_torchvision_model
from src.utils.reproducibility import get_device
from src.utils.save_metrics import save_experiment


def create_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
    model_cfg: dict[str, Any] | None = None,
) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    model_cfg = model_cfg or {}
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(model_cfg.get("learning_rate", train_cfg["learning_rate"]))
    weight_decay = float(train_cfg["weight_decay"])
    if train_cfg.get("optimizer", "adamw").lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, device, desc: str = "train") -> float:
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(loader), 1)


def train_torchvision(model_name: str, config: dict[str, Any]) -> dict[str, float]:
    device = get_device(config.get("device", "auto"))
    processed_dir = Path(config["data"]["processed_dir"])
    image_size = int(config["data"]["image_size"])
    train_cfg = config["training"]
    model_cfg = config["models"][model_name]

    train_dataset = GTSDBDetectionDataset(
        processed_dir / "manifest_train.csv",
        image_size=image_size,
        train=True,
    )
    val_dataset = GTSDBDetectionDataset(
        processed_dir / "manifest_val.csv",
        image_size=image_size,
        train=False,
    )

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

    model = create_torchvision_model(model_name, config).to(device)
    optimizer = create_optimizer(model, config, model_cfg)

    save_dir = Path(train_cfg["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.csv"
    history_path.write_text("epoch,train_loss,map,map50,precision,recall,f1\n", encoding="utf-8")

    best_map = -1.0
    best_metrics: dict[str, float] = {}
    total_epochs = int(train_cfg["epochs"])
    for epoch in range(1, total_epochs + 1):
        print(f"\n=== {model_name}: epoch {epoch}/{total_epochs} ===")
        loss = train_one_epoch(model, train_loader, optimizer, device, desc=f"train {epoch}/{total_epochs}")
        metrics = evaluate_torchvision(model, val_loader, device, desc=f"eval {epoch}/{total_epochs}")
        history_path.write_text(
            history_path.read_text(encoding="utf-8")
            + f"{epoch},{loss:.6f},{metrics['map']:.6f},{metrics['map50']:.6f},"
            + f"{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f}\n",
            encoding="utf-8",
        )
        if metrics["map"] > best_map:
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
