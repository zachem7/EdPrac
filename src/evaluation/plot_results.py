from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


METRIC_PLOTS = [
    ("train_loss", "Train loss"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
    ("map50", "mAP50"),
    ("map", "mAP50-95"),
]

METRIC_LABELS = {
    "metric.map": "mAP50-95",
    "metric.map50": "mAP50",
    "metric.precision": "Precision",
    "metric.recall": "Recall",
    "metric.f1": "F1-score",
}


def smooth(values: pd.Series, window: int = 3) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


def prepare_history_frame(history_path: Path) -> pd.DataFrame:
    df = pd.read_csv(history_path)
    df.columns = [column.strip() for column in df.columns]

    normalized = pd.DataFrame()
    if "epoch" in df.columns:
        normalized["epoch"] = df["epoch"]
    else:
        normalized["epoch"] = range(1, len(df) + 1)

    # Regular project histories: epoch,train_loss,map,map50,precision,recall,f1.
    for column in ["train_loss", "map", "map50", "precision", "recall", "f1"]:
        if column in df.columns:
            normalized[column] = pd.to_numeric(df[column], errors="coerce")

    # Ultralytics YOLO history: results.csv with columns like metrics/mAP50(B).
    yolo_loss_columns = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    available_loss_columns = [column for column in yolo_loss_columns if column in df.columns]
    if "train_loss" not in normalized and available_loss_columns:
        normalized["train_loss"] = df[available_loss_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    yolo_columns = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map",
    }
    for source, target in yolo_columns.items():
        if target not in normalized and source in df.columns:
            normalized[target] = pd.to_numeric(df[source], errors="coerce")

    if "f1" not in normalized and {"precision", "recall"}.issubset(normalized.columns):
        precision = normalized["precision"]
        recall = normalized["recall"]
        normalized["f1"] = 2 * precision * recall / (precision + recall).clip(lower=1e-8)

    return normalized


def plot_model_history(model_name: str, history_path: Path, plots_dir: Path) -> None:
    df = prepare_history_frame(history_path)
    available = [(column, title) for column, title in METRIC_PLOTS if column in df.columns]
    if not available:
        return

    model_plots_dir = plots_dir / "training_curves"
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes_flat = axes.flatten()

    for axis, (column, title) in zip(axes_flat, available):
        series = df[column]
        axis.plot(df["epoch"], series, marker="o", label="значения")
        axis.plot(df["epoch"], smooth(series), linestyle=":", linewidth=2.5, label="сглаживание")
        axis.set_title(title)
        axis.set_xlabel("Эпоха")
        axis.grid(alpha=0.25)

    for axis in axes_flat[len(available) :]:
        axis.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(f"{model_name}: метрики обучения", y=0.99, fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(model_plots_dir / f"{model_name}_training_metrics.png", dpi=200)
    plt.close(fig)


def existing_path(path_text: str | float | None) -> Path | None:
    if path_text is None or pd.isna(path_text):
        return None

    path_value = str(path_text).strip()
    if not path_value:
        return None

    path = Path(path_value)
    if path.exists():
        return path

    normalized = Path(path_value.replace("\\", "/"))
    if normalized.exists():
        return normalized

    return None


def add_candidate(candidates: list[Path], path: Path | None) -> None:
    if path is not None and path not in candidates:
        candidates.append(path)


def history_candidates_for_model(model_name: str, config: dict[str, Any], experiments_df: pd.DataFrame) -> list[Path]:
    logs_dir = Path(config["training"]["save_dir"])
    model_dir = logs_dir / model_name
    candidates: list[Path] = []

    add_candidate(candidates, model_dir / "history.csv")
    add_candidate(candidates, model_dir / "results.csv")

    if "model" in experiments_df.columns:
        model_runs = experiments_df[experiments_df["model"] == model_name]
        if "timestamp" in model_runs.columns:
            model_runs = model_runs.sort_values("timestamp", ascending=False)

        for _, row in model_runs.iterrows():
            history_path = existing_path(row.get("metric.history"))
            add_candidate(candidates, history_path)

            train_dir = existing_path(row.get("metric.train_results_dir"))
            if train_dir is not None:
                add_candidate(candidates, train_dir / "results.csv")
                add_candidate(candidates, train_dir / "history.csv")

    for history_path in sorted(logs_dir.glob(f"{model_name}*/history.csv")):
        add_candidate(candidates, history_path)

    for results_path in sorted(logs_dir.glob(f"{model_name}*/results.csv")):
        add_candidate(candidates, results_path)

    for results_path in sorted(logs_dir.rglob("results.csv")):
        if model_name in str(results_path):
            add_candidate(candidates, results_path)

    return candidates


def plot_training_histories(config: dict[str, Any], plots_dir: Path, experiments_df: pd.DataFrame) -> None:
    logs_dir = Path(config["training"]["save_dir"])
    if not logs_dir.exists():
        return

    for model_name in config["models"]:
        for history_path in history_candidates_for_model(model_name, config, experiments_df):
            if history_path.exists():
                plot_model_history(model_name, history_path, plots_dir)
                break


def plot_results(config: dict[str, Any]) -> None:
    metrics_path = Path(config["results"]["metrics_csv"])
    plots_dir = Path(config["results"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Файл с метриками не найден: {metrics_path}")

    df = pd.read_csv(metrics_path)
    metric_columns = ["metric.map", "metric.map50", "metric.precision", "metric.recall", "metric.f1"]
    available = [column for column in metric_columns if column in df.columns]

    latest = df.sort_values("timestamp").groupby("model", as_index=False).tail(1)

    for column in available:
        label = METRIC_LABELS.get(column, column.replace("metric.", ""))
        plt.figure(figsize=(10, 5))
        plt.bar(latest["model"], latest[column])
        plt.title(f"Сравнение моделей по {label}")
        plt.xlabel("Модель")
        plt.ylabel(label)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{column.replace('metric.', '')}_comparison.png", dpi=200)
        plt.close()

    plot_training_histories(config, plots_dir, df)

    print(f"Графики сохранены в {plots_dir}")
