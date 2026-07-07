from typing import Any

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn


def build_faster_rcnn(config: dict[str, Any]):
    num_classes = int(config["data"]["num_classes"]) + 1
    return fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
