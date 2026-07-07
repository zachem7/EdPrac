from typing import Any

from ultralytics import YOLO


def build_yolo(model_name: str, config: dict[str, Any]) -> YOLO:
    return YOLO(config["models"][model_name]["weights"])

