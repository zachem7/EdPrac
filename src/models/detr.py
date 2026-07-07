from typing import Any

from transformers import DetrConfig, DetrForObjectDetection


def build_detr(model_name: str, config: dict[str, Any]) -> DetrForObjectDetection:
    model_cfg = config["models"][model_name]
    train_cfg = config["training"]
    num_labels = int(config["data"]["num_classes"])
    if bool(train_cfg.get("pretrained", False)):
        return DetrForObjectDetection.from_pretrained(
            model_cfg.get("pretrained_name", "facebook/detr-resnet-50"),
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    detr_config = DetrConfig(
        num_labels=num_labels,
        num_queries=int(model_cfg.get("num_queries", 100)),
        use_pretrained_backbone=False,
    )
    return DetrForObjectDetection(detr_config)
