from __future__ import annotations

import argparse

from src.utils.config import ensure_dirs, load_config
from src.utils.reproducibility import set_seed


def train_model(model_name: str, config: dict) -> dict:
    model_type = config["models"][model_name]["type"]
    if model_type == "yolo":
        from src.training.train_yolo import train_yolo

        return train_yolo(model_name, config)
    if model_type == "torchvision":
        from src.training.train_torchvision import train_torchvision

        return train_torchvision(model_name, config)
    if model_type == "efficientdet":
        from src.training.train_efficientdet import train_efficientdet

        return train_efficientdet(model_name, config)
    if model_type == "detr":
        from src.training.train_detr import train_detr

        return train_detr(model_name, config)
    raise ValueError(f"Неизвестный тип модели: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GTSDB traffic sign detection pipeline")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "analyze",
            "train",
            "train-all",
            "plot",
        ],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="yolov8n")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)
    set_seed(int(config["seed"]))

    if args.command == "prepare":
        from src.dataset.prepare_gtsdb import prepare_gtsdb

        prepare_gtsdb(config)
        return

    if args.command == "analyze":
        from src.dataset.analyze_gtsdb import analyze_gtsdb

        analyze_gtsdb(config)
        return

    if args.command == "train":
        metrics = train_model(args.model, config)
        print(metrics)
        return

    if args.command == "train-all":
        for model_name in config["models"]:
            print(f"\n=== Training {model_name} ===")
            metrics = train_model(model_name, config)
            print(metrics)
        return

    if args.command == "plot":
        from src.evaluation.plot_results import plot_results

        plot_results(config)


if __name__ == "__main__":
    main()
