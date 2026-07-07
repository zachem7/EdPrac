from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.gtsdb_dataset import GTSDBDetectionDataset, detection_collate
from src.evaluation.evaluate_efficientdet import (
    evaluate_efficientdet,
    get_efficientdet_eval_settings,
)
from src.models.efficientdet import build_efficientdet_predict, build_efficientdet_train
from src.training.train_torchvision import create_optimizer
from src.utils.reproducibility import get_device
from src.utils.save_metrics import save_experiment


HISTORY_HEADER = "epoch,train_loss,map,map50,precision,recall,f1\n"


def build_efficientdet_targets(targets: list[dict], image_size: int, device) -> dict:
    max_boxes = max(1, max(len(target["boxes"]) for target in targets))
    batch_size = len(targets)

    boxes = torch.zeros(
        (batch_size, max_boxes, 4),
        dtype=torch.float32,
        device=device,
    )
    classes = torch.full(
        (batch_size, max_boxes),
        -1,
        dtype=torch.float32,
        device=device,
    )

    for index, target in enumerate(targets):
        count = len(target["boxes"])
        if count == 0:
            continue

        xyxy_boxes = target["boxes"].to(device)
        boxes[index, :count] = xyxy_boxes[:, [1, 0, 3, 2]]
        classes[index, :count] = target["labels"].to(device).float()

    return {
        "bbox": boxes,
        "cls": classes,
        "img_size": torch.full(
            (batch_size, 2),
            image_size,
            dtype=torch.float32,
            device=device,
        ),
        "img_scale": torch.ones(
            (batch_size,),
            dtype=torch.float32,
            device=device,
        ),
    }


def make_loader(dataset, train_cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=detection_collate,
    )


def append_history_row(history_path: Path, epoch: int, train_loss: float, metrics: dict[str, Any]) -> None:
    with history_path.open("a", encoding="utf-8") as file:
        file.write(
            f"{epoch},"
            f"{train_loss:.6f},"
            f"{metrics['map']:.6f},"
            f"{metrics['map50']:.6f},"
            f"{metrics['precision']:.6f},"
            f"{metrics['recall']:.6f},"
            f"{metrics['f1']:.6f}\n"
        )


def train_one_epoch_efficientdet(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device,
    image_size: int,
    desc: str,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(loader, desc=desc, leave=False):
        inputs = torch.stack([image.to(device) for image in images])
        eff_targets = build_efficientdet_targets(targets, image_size, device)

        loss_output = model(inputs, eff_targets)
        loss = loss_output["loss"] if isinstance(loss_output, dict) else loss_output

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(loader), 1)


def train_efficientdet(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    device = get_device(config.get("device", "auto"))

    processed_dir = Path(config["data"]["processed_dir"])
    model_cfg = config["models"][model_name]
    train_cfg = config["training"]

    image_size = int(model_cfg.get("image_size", config["data"]["image_size"]))
    num_classes = int(config["data"]["num_classes"])
    total_epochs = int(train_cfg["epochs"])

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

    train_loader = make_loader(train_dataset, train_cfg, shuffle=True)
    val_loader = make_loader(val_dataset, train_cfg, shuffle=False)

    model = build_efficientdet_train(model_name, config).to(device)
    optimizer = create_optimizer(model, config, model_cfg)

    score_threshold, top_k = get_efficientdet_eval_settings(config, model_name)

    save_dir = Path(train_cfg["save_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history_path = save_dir / "history.csv"
    history_path.write_text(HISTORY_HEADER, encoding="utf-8")

    best_score = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, Any] = {}

    for epoch in range(1, total_epochs + 1):
        print(f"\n=== {model_name}: epoch {epoch}/{total_epochs} ===")

        train_loss = train_one_epoch_efficientdet(
            model,
            train_loader,
            optimizer,
            device,
            image_size,
            desc=f"train {epoch}/{total_epochs}",
        )

        torch.save(model.state_dict(), save_dir / "last.pth")

        predict_model = build_efficientdet_predict(model_name, config).to(device)
        predict_model.load_state_dict(
            torch.load(save_dir / "last.pth", map_location=device),
            strict=False,
        )

        metrics = evaluate_efficientdet(
            predict_model,
            val_loader,
            device,
            image_size=image_size,
            num_classes=num_classes,
            score_threshold=score_threshold,
            top_k=top_k,
            desc=f"eval {epoch}/{total_epochs}",
        )

        append_history_row(history_path, epoch, train_loss, metrics)

        score = (
            metrics["map50"],
            metrics["map"],
            metrics["f1"],
        )
        if score >= best_score:
            best_score = score
            best_metrics = dict(metrics)
            torch.save(model.state_dict(), save_dir / "best.pth")

    best_metrics["checkpoint"] = str(save_dir / "best.pth")
    best_metrics["history"] = str(history_path)

    save_experiment(
        model_name=model_name,
        hyperparameters={**train_cfg, **model_cfg},
        metrics=best_metrics,
        csv_path=config["results"]["metrics_csv"],
        jsonl_path=config["results"]["metrics_jsonl"],
    )

    return best_metrics
