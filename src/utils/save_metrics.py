import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_dict(value, name))
        else:
            result[name] = value
    return result


def save_experiment(
    model_name: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    csv_path: str | Path,
    jsonl_path: str | Path,
) -> None:
    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        **flatten_dict({"hp": hyperparameters}),
        **flatten_dict({"metric": metrics}),
    }

    rows: list[dict[str, Any]] = []
    fieldnames = list(record.keys())
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

    for key in record:
        if key not in fieldnames:
            fieldnames.append(key)

    rows.append(record)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(jsonl_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
