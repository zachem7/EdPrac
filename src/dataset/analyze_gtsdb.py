from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _read_manifest(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"manifest_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала запусти подготовку: python main.py prepare --config configs/default.yaml"
        )
    return pd.read_csv(path)


def analyze_gtsdb(config: dict[str, Any]) -> None:
    processed_dir = Path(config["data"]["processed_dir"])
    plots_dir = Path(config["results"]["plots_dir"])
    metrics_dir = Path(config["results"]["metrics_csv"]).parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_names = config["data"]["class_names"]
    frames = []
    split_stats = []

    for split in ["train", "val", "test"]:
        df = _read_manifest(processed_dir, split)
        df["split"] = split
        frames.append(df)
        image_count = df["image_path"].nunique()
        annotation_count = len(df)
        split_stats.append(
            {
                "split": split,
                "images": image_count,
                "annotations": annotation_count,
                "avg_boxes_per_image": annotation_count / max(image_count, 1),
            }
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df["class_name"] = all_df["class_id"].map(lambda class_id: class_names[int(class_id)])
    all_df["bbox_width"] = all_df["x2"] - all_df["x1"]
    all_df["bbox_height"] = all_df["y2"] - all_df["y1"]
    all_df["bbox_area_ratio"] = (all_df["bbox_width"] * all_df["bbox_height"]) / (
        all_df["width"] * all_df["height"]
    )

    split_stats_df = pd.DataFrame(split_stats)
    class_distribution = (
        all_df.groupby(["class_id", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "annotations"})
        .sort_values("annotations", ascending=False)
    )

    split_stats_df.to_csv(metrics_dir / "dataset_split_stats.csv", index=False)
    class_distribution.to_csv(metrics_dir / "dataset_class_distribution.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(class_distribution["class_name"], class_distribution["annotations"])
    plt.title("Распределение классов GTSDB")
    plt.xlabel("Класс")
    plt.ylabel("Количество объектов")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plots_dir / "dataset_class_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(all_df["bbox_area_ratio"], bins=30)
    plt.title("Доля площади bounding box от площади изображения")
    plt.xlabel("Площадь bounding box / площадь изображения")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig(plots_dir / "dataset_bbox_area_ratio.png", dpi=200)
    plt.close()

    print(f"Анализ датасета сохранён в {metrics_dir} и {plots_dir}")
