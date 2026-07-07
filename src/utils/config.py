from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dirs(config: dict[str, Any]) -> None:
    paths = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["training"]["save_dir"],
        config["results"]["plots_dir"],
        Path(config["results"]["metrics_csv"]).parent,
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

