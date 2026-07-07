from __future__ import annotations

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def compute_map_from_batches(pred_batches: list[list[dict]], target_batches: list[list[dict]]) -> dict[str, float]:
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.warn_on_many_detections = False
    for preds, targets in zip(pred_batches, target_batches):
        metric.update(preds, targets)

    result = metric.compute()
    precision = float(result.get("map_50", torch.tensor(0.0)))
    recall = float(result.get("mar_100", torch.tensor(0.0)))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "map": float(result["map"]),
        "map50": float(result["map_50"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def target_to_cpu(target: dict) -> dict:
    return {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu(),
    }

