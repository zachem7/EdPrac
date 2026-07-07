from __future__ import annotations

import torch
from tqdm import tqdm

from src.evaluation.detection_metrics import compute_map_from_batches, target_to_cpu


@torch.no_grad()
def evaluate_torchvision(model, loader, device, desc: str = "eval") -> dict[str, float]:
    model.eval()

    total_pred = 0
    total_gt = 0
    pred_batches = []
    target_batches = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = [image.to(device) for image in images]
        outputs = model(images)

        preds = []
        gts = []
        for output, target in zip(outputs, targets):
            pred = {
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            }
            gt = target_to_cpu(target)
            preds.append(pred)
            gts.append(gt)
            total_pred += len(pred["boxes"])
            total_gt += len(gt["boxes"])

        pred_batches.append(preds)
        target_batches.append(gts)

    metrics = compute_map_from_batches(pred_batches, target_batches)
    metrics.update(
        {
            "predictions": float(total_pred),
            "ground_truth": float(total_gt),
        }
    )
    return metrics
