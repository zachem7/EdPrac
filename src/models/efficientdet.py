from __future__ import annotations

from typing import Any

from effdet import create_model


def use_pretrained_weights(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("pretrained", False))


def build_efficientdet_train(model_name: str, config: dict[str, Any]):
    model_cfg = config["models"][model_name]

    return create_model(
        model_cfg["architecture"],
        bench_task="train",
        num_classes=int(config["data"]["num_classes"]),
        pretrained=use_pretrained_weights(config),
        pretrained_backbone=use_pretrained_weights(config),
        bench_labeler=True,
    )


def build_efficientdet_predict(model_name: str, config: dict[str, Any]):
    model_cfg = config["models"][model_name]

    return create_model(
        model_cfg["architecture"],
        bench_task="predict",
        num_classes=int(config["data"]["num_classes"]),
        pretrained=False,
        pretrained_backbone=False,
    )
