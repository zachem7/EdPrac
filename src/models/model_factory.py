from __future__ import annotations

from typing import Any

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def create_torchvision_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    num_classes_with_background = int(config["data"]["num_classes"]) + 1
    model_cfg = config["models"][model_name]
    train_cfg = config["training"]
    architecture = model_cfg["architecture"]
    use_pretrained = bool(train_cfg.get("pretrained", False))
    backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
    weights = None

    if architecture == "fasterrcnn_mobilenet_v3_large_fpn":
        return fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes_with_background,
        )

    if architecture == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(
            weights=weights,
            weights_backbone=backbone_weights,
            num_classes=num_classes_with_background,
        )

    raise ValueError(f"Неизвестная torchvision-архитектура: {architecture}")
