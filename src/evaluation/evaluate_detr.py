from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from src.evaluation.detection_metrics import compute_map_from_batches, target_to_cpu


def cxcywh_normalized_to_xyxy(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)

    x1 = (cx - width / 2) * image_size
    y1 = (cy - height / 2) * image_size
    x2 = (cx + width / 2) * image_size
    y2 = (cy + height / 2) * image_size

    return torch.stack([x1, y1, x2, y2], dim=-1)


def get_detr_eval_settings(config: dict[str, Any], model_name: str) -> tuple[float, int]:
    model_cfg = config["models"][model_name]
    eval_cfg = config["evaluation"]

    score_threshold = float(
        model_cfg.get("score_threshold", min(float(eval_cfg["score_threshold"]), 0.01))
    )
    top_k = int(model_cfg.get("top_k", 100))

    return score_threshold, top_k


def decode_detr_prediction(
    image_scores: torch.Tensor,
    image_labels: torch.Tensor,
    image_boxes: torch.Tensor,
    image_size: int,
    score_threshold: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    boxes = image_boxes.clone()
    scores = image_scores.clone()
    labels = image_labels.long() + 1

    finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    boxes = boxes[finite]
    scores = scores[finite]
    labels = labels[finite]

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
        boxes = boxes[valid_boxes]
        scores = scores[valid_boxes]
        labels = labels[valid_boxes]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


@torch.no_grad()
def evaluate_detr(
    model,
    loader,
    device,
    image_size: int,
    score_threshold: float,
    top_k: int,
    desc: str = "eval",
) -> dict[str, Any]:
    model.eval()

    pred_batches = []
    target_batches = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        pixel_values = torch.stack([image.to(device) for image in images])
        outputs = model(pixel_values=pixel_values)

        probabilities = outputs.logits.softmax(-1)
        scores, labels = probabilities[..., :-1].max(-1)
        boxes = cxcywh_normalized_to_xyxy(
            outputs.pred_boxes.detach().cpu(),
            image_size,
        )

        batch_preds = []
        batch_targets = []

        for image_scores, image_labels, image_boxes, target in zip(
            scores.cpu(),
            labels.cpu(),
            boxes,
            targets,
        ):
            pred = decode_detr_prediction(
                image_scores=image_scores,
                image_labels=image_labels,
                image_boxes=image_boxes,
                image_size=image_size,
                score_threshold=score_threshold,
                top_k=top_k,
            )

            batch_preds.append(pred)
            batch_targets.append(target_to_cpu(target))

        pred_batches.append(batch_preds)
        target_batches.append(batch_targets)

    return compute_map_from_batches(pred_batches, target_batches)
