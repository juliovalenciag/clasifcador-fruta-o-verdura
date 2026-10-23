from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    datasets_config: Path
    classes: Path
    label_mapping: Path
    manifest: Path
    deduplicated_manifest: Path
    duplicate_report: Path
    split_manifest: Path
    audit_report: Path
    excluded_images: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            datasets_config=root / "configs" / "datasets.yaml",
            classes=root / "configs" / "classes.csv",
            label_mapping=root / "configs" / "label_mapping.csv",
            manifest=root / "data" / "interim" / "manifest.csv",
            deduplicated_manifest=root / "data" / "interim" / "manifest_deduplicated.csv",
            duplicate_report=root / "data" / "interim" / "duplicate_candidates.csv",
            split_manifest=root / "data" / "processed" / "dataset.csv",
            audit_report=root / "data" / "interim" / "data_report.md",
            excluded_images=root / "configs" / "excluded_images.csv",
        )


def load_dataset_config(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("configs/datasets.yaml debe contener una lista 'datasets'.")
    ids = [item.get("id") for item in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("Los identificadores de datasets deben ser únicos.")
    return datasets

