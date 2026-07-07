from typing import Any

from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def build_ssd(config: dict[str, Any]):
    num_classes = int(config["data"]["num_classes"]) + 1
    return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
