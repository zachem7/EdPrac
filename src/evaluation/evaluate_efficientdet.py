from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from src.evaluation.detection_metrics import compute_map_from_batches, target_to_cpu


def get_efficientdet_eval_settings(config: dict[str, Any], model_name: str) -> tuple[float, int]:
    model_cfg = config["models"][model_name]
    eval_cfg = config["evaluation"]

    score_threshold = float(
        model_cfg.get("score_threshold", min(float(eval_cfg["score_threshold"]), 0.05))
    )
    top_k = int(model_cfg.get("top_k", 50))

    return score_threshold, top_k


def decode_prediction(
    detection: torch.Tensor,
    image_size: int,
    num_classes: int,
    score_threshold: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    boxes = detection[:, 0:4].clone()
    scores = detection[:, 4]
    labels = detection[:, 5].long() + 1

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_size)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_size)

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(scores) > top_k:
        indices = scores.argsort(descending=True)[:top_k]
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

    if len(boxes) > 0:
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid_labels = (labels >= 1) & (labels <= num_classes)
        valid = valid_boxes & valid_labels

        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


@torch.no_grad()
def evaluate_efficientdet(
    model,
    loader,
    device,
    image_size: int,
    num_classes: int,
    score_threshold: float,
    top_k: int,
    desc: str = "eval",
) -> dict[str, Any]:
    model.eval()

    pred_batches = []
    target_batches = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        inputs = torch.stack([image.to(device) for image in images])
        detections = model(inputs).detach().cpu()

        batch_preds = []
        batch_targets = []

        for detection, target in zip(detections, targets):
            pred = decode_prediction(
                detection=detection,
                image_size=image_size,
                num_classes=num_classes,
                score_threshold=score_threshold,
                top_k=top_k,
            )
            gt = target_to_cpu(target)

            batch_preds.append(pred)
            batch_targets.append(gt)

        pred_batches.append(batch_preds)
        target_batches.append(batch_targets)

    return compute_map_from_batches(pred_batches, target_batches)
