from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import imagehash
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .config import ProjectPaths, load_dataset_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MANIFEST_COLUMNS = [
    "image_id",
    "source",
    "dataset_role",
    "path",
    "relative_path",
    "original_label",
    "normalized_original_label",
    "class_id",
    "mapping_status",
    "original_split",
    "width",
    "height",
    "mode",
    "format",
    "file_size",
    "sha256",
    "phash",
    "valid",
    "error",
]


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return value.strip("_")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_split(relative_path: Path, aliases: dict[str, list[str]]) -> str:
    lookup = {
        normalize_label(alias): canonical
        for canonical, values in (aliases or {}).items()
        for alias in values
    }
    for part in relative_path.parts[:-1]:
        canonical = lookup.get(normalize_label(part))
        if canonical:
            return canonical
    return "unspecified"


def _read_image_metadata(path: Path) -> dict[str, Any]:
    base = {
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "file_size": path.stat().st_size,
        "sha256": "",
        "phash": "",
        "valid": False,
        "error": "",
    }
    try:
        digest = _sha256(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width < 16 or image.height < 16:
                raise ValueError(
                    f"imagen demasiado pequeña para clasificar: {image.width}x{image.height}"
                )
            base.update(
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image.format or path.suffix.lstrip(".").upper(),
                sha256=digest,
                phash=str(imagehash.phash(image.convert("RGB"))),
                valid=True,
            )
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _load_mapping(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"source", "original_label", "class_id", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en label_mapping.csv: {sorted(missing)}")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for record in frame.to_dict("records"):
        key = (record["source"], normalize_label(record["original_label"]))
        if key in result:
            raise ValueError(f"Mapeo duplicado para {key}.")
        result[key] = record
    return result


def _validate_classes(paths: ProjectPaths, mapping: dict[tuple[str, str], dict[str, str]]) -> None:
    classes = pd.read_csv(paths.classes, dtype=str).fillna("")
    required = {"class_id", "display_name_es"}
    missing = required.difference(classes.columns)
    if missing:
        raise ValueError(f"Faltan columnas en classes.csv: {sorted(missing)}")
    ids = set(classes["class_id"])
    duplicates = classes[classes["class_id"].duplicated()]["class_id"].tolist()
    if duplicates:
        raise ValueError(f"IDs duplicados en classes.csv: {duplicates}")
    if ids != {"fruta", "verdura"}:
        raise ValueError("classes.csv debe declarar exactamente las clases 'fruta' y 'verdura'.")
    referenced = {record["class_id"] for record in mapping.values() if record["class_id"]}
    missing_ids = sorted(referenced.difference(ids))
    if missing_ids:
        raise ValueError(f"El mapeo referencia IDs ausentes de classes.csv: {missing_ids}")


def _iter_images(root: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_manifest(paths: ProjectPaths) -> pd.DataFrame:
    datasets = load_dataset_config(paths.datasets_config)
    mapping = _load_mapping(paths.label_mapping)
    _validate_classes(paths, mapping)
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        if not dataset.get("enabled", True):
            continue
        source = str(dataset["id"])
        dataset_root = (paths.root / str(dataset["path"])).resolve()
        if not dataset_root.exists():
            print(f"Aviso: se omite {source}; no existe {dataset_root}")
            continue
        images = list(_iter_images(dataset_root))
        print(f"Inventariando {source}: {len(images)} imágenes")
        workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            metadata_stream = executor.map(_read_image_metadata, images)
            records = zip(images, metadata_stream)
            progress = tqdm(records, total=len(images), desc=source, unit="img")
            for image_path, metadata in progress:
                relative = image_path.relative_to(dataset_root)
                original_label = image_path.parent.name
                normalized = normalize_label(original_label)
                mapped = mapping.get((source, normalized), {})
                stable_key = f"{source}:{relative.as_posix()}".encode("utf-8")
                rows.append(
                    {
                        "image_id": hashlib.sha1(stable_key).hexdigest(),
                        "source": source,
                        "dataset_role": dataset.get("role", "trainable"),
                        "path": str(image_path),
                        "relative_path": relative.as_posix(),
                        "original_label": original_label,
                        "normalized_original_label": normalized,
                        "class_id": mapped.get("class_id", ""),
                        "mapping_status": mapped.get("status", "unmapped"),
                        "original_split": _detect_split(relative, dataset.get("split_aliases", {})),
                        **metadata,
                    }
                )

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.manifest, index=False)
    print(f"Manifiesto guardado en {paths.manifest} ({len(frame)} registros).")
    return frame

