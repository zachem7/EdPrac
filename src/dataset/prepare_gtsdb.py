from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


GTSDB_COLUMNS = [
    "filename",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "class_id",
]


def find_annotations_file(raw_dir: Path, configured_file: str | None) -> Path:
    if configured_file:
        path = Path(configured_file)
        return path if path.is_absolute() else raw_dir / path

    candidates = []
    candidates.extend(raw_dir.rglob("gt.txt"))
    candidates.extend(raw_dir.rglob("GT*.csv"))
    candidates.extend(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Файл аннотаций GTSDB не найден в {raw_dir}. "
        )
    return candidates[0]


def locate_image(raw_dir: Path, filename: str, annotation_dir: Path | None = None) -> Path:
    search_roots = []
    if annotation_dir is not None:
        search_roots.append(annotation_dir)
    search_roots.append(raw_dir)

    for root in search_roots:
        direct = root / filename
        if direct.exists():
            return direct

    matches = list(raw_dir.rglob(filename))
    if matches:
        if annotation_dir is not None:
            for match in matches:
                if match.parent == annotation_dir:
                    return match
        return matches[0]
    raise FileNotFoundError(f"Изображение {filename} не найдено в {raw_dir}.")


def add_image_sizes(df: pd.DataFrame, raw_dir: Path, annotation_dir: Path) -> pd.DataFrame:
    widths: dict[str, int] = {}
    heights: dict[str, int] = {}
    for filename in df["filename"].unique():
        image_path = locate_image(raw_dir, filename, annotation_dir)
        with Image.open(image_path) as image:
            widths[filename], heights[filename] = image.size
    df["width"] = df["filename"].map(widths)
    df["height"] = df["filename"].map(heights)
    return df


def read_gtsdb_annotations(annotation_path: Path, raw_dir: Path) -> pd.DataFrame:
    if annotation_path.suffix.lower() == ".txt":
        df = pd.read_csv(annotation_path, sep=";", header=None)
        if len(df.columns) < 6:
            raise ValueError("gt.txt должен иметь формат: filename;x1;y1;x2;y2;class_id")
        df = df.iloc[:, :6]
        df.columns = ["filename", "x1", "y1", "x2", "y2", "class_id"]
        df = add_image_sizes(df, raw_dir, annotation_path.parent)
        df = df[["filename", "width", "height", "x1", "y1", "x2", "y2", "class_id"]]
    else:
        df = pd.read_csv(annotation_path, sep=";")
        if len(df.columns) >= 8:
            df = df.iloc[:, :8]
            df.columns = GTSDB_COLUMNS
        else:
            df = pd.read_csv(annotation_path, sep=";", header=None)
            if len(df.columns) < 6:
                raise ValueError("CSV GTSDB должен содержать 6 или 8 столбцов.")
            df = df.iloc[:, :6]
            df.columns = ["filename", "x1", "y1", "x2", "y2", "class_id"]
            df = add_image_sizes(df, raw_dir, annotation_path.parent)
            df = df[["filename", "width", "height", "x1", "y1", "x2", "y2", "class_id"]]

    if len(df.columns) >= 8:
        df = df.iloc[:, :8]
        df.columns = GTSDB_COLUMNS

    df["filename"] = df["filename"].astype(str)
    for column in ["width", "height", "x1", "y1", "x2", "y2", "class_id"]:
        df[column] = df[column].astype(int)
    return df


def split_images(filenames: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[str]]:
    names = filenames[:]
    random.Random(seed).shuffle(names)
    train_end = int(len(names) * train_ratio)
    val_end = train_end + int(len(names) * val_ratio)
    return {
        "train": set(names[:train_end]),
        "val": set(names[train_end:val_end]),
        "test": set(names[val_end:]),
    }


def yolo_line(row: pd.Series) -> str:
    x_center = ((row.x1 + row.x2) / 2) / row.width
    y_center = ((row.y1 + row.y2) / 2) / row.height
    box_width = (row.x2 - row.x1) / row.width
    box_height = (row.y2 - row.y1) / row.height
    return f"{row.class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def save_yolo_image(source_image: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=95)


def prepare_gtsdb(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    annotations_file = find_annotations_file(raw_dir, data_cfg.get("annotations_csv"))
    df = read_gtsdb_annotations(annotations_file, raw_dir)

    filenames = sorted(df["filename"].unique().tolist())
    splits = split_images(
        filenames,
        float(data_cfg["train_ratio"]),
        float(data_cfg["val_ratio"]),
        int(config["seed"]),
    )

    yolo_root = processed_dir / "yolo"
    torch_root = processed_dir / "torchvision"
    for split in splits:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (torch_root / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split, split_names in splits.items():
        split_df = df[df["filename"].isin(split_names)]
        for filename, group in split_df.groupby("filename"):
            source_image = locate_image(raw_dir, filename, annotations_file.parent)
            yolo_image = yolo_root / "images" / split / f"{source_image.stem}.jpg"
            torch_image = torch_root / split / source_image.name
            save_yolo_image(source_image, yolo_image)
            shutil.copy2(source_image, torch_image)

            label_path = yolo_root / "labels" / split / f"{yolo_image.stem}.txt"
            label_path.write_text("\n".join(yolo_line(row) for _, row in group.iterrows()), encoding="utf-8")

            for _, row in group.iterrows():
                manifest_rows[split].append(
                    {
                        "image_path": str(torch_image),
                        "filename": filename,
                        "width": int(row.width),
                        "height": int(row.height),
                        "x1": int(row.x1),
                        "y1": int(row.y1),
                        "x2": int(row.x2),
                        "y2": int(row.y2),
                        "class_id": int(row.class_id),
                    }
                )

    for split, rows in manifest_rows.items():
        with open(processed_dir / f"manifest_{split}.csv", "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["image_path"])
            writer.writeheader()
            writer.writerows(rows)

    yaml_text = [
        f"path: {yolo_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {data_cfg['num_classes']}",
        "names:",
    ]
    for index, name in enumerate(data_cfg["class_names"]):
        yaml_text.append(f"  {index}: {name}")
    (processed_dir / "gtsdb_yolo.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")

    print(f"Данные подготовлены в {processed_dir}")
