from __future__ import annotations

import hashlib
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from .config import ProjectPaths, load_dataset_config
from .manifest import IMAGE_EXTENSIONS, normalize_label


# Nombres canónicos de los 63 IDs declarados en data.yaml del commit fijado.
# Se usa el primer sinónimo y se unifican las etiquetas repetidas de fresa/tomate.
LVIS_CLASS_NAMES = (
    "almond",
    "apple",
    "apricot",
    "artichoke",
    "asparagus",
    "avocado",
    "banana",
    "bean curd tofu",
    "bell pepper",
    "blackberry",
    "blueberry",
    "broccoli",
    "brussels sprouts",
    "cantaloupe",
    "carrot",
    "cauliflower",
    "cayenne pepper",
    "celery",
    "cherry",
    "chickpea",
    "chili pepper",
    "clementine",
    "coconut",
    "corn",
    "cucumber",
    "date",
    "eggplant",
    "fig",
    "garlic",
    "ginger",
    "strawberry",
    "gourd",
    "grape",
    "green bean",
    "green onion",
    "tomato",
    "kiwi",
    "lemon",
    "lettuce",
    "lime",
    "mandarin orange",
    "melon",
    "mushroom",
    "onion",
    "orange",
    "papaya",
    "pea",
    "peach",
    "pear",
    "persimmon",
    "pickle",
    "pineapple",
    "potato",
    "prune",
    "pumpkin",
    "radish",
    "raspberry",
    "strawberry",
    "sweet potato",
    "tomato",
    "turnip",
    "watermelon",
    "zucchini",
)


def _dataset_entry(paths: ProjectPaths, dataset_id: str) -> dict[str, object]:
    for dataset in load_dataset_config(paths.datasets_config):
        if dataset.get("id") == dataset_id:
            return dataset
    raise ValueError(f"Dataset desconocido: {dataset_id!r}.")


def _read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        values = raw_line.split()
        if len(values) != 5:
            continue
        try:
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            continue
        if not 0 <= class_id < len(LVIS_CLASS_NAMES):
            continue
        if width <= 0 or height <= 0:
            continue
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def prepare_lvis_dataset(
    raw_root: Path,
    output_root: Path,
    margin_ratio: float = 0.15,
    min_object_size: int = 24,
) -> dict[str, object]:
    """Convierte anotaciones YOLO en recortes para clasificación de imágenes."""

    if not 0 <= margin_ratio <= 1:
        raise ValueError("margin_ratio debe estar entre 0 y 1.")
    if min_object_size < 1:
        raise ValueError("min_object_size debe ser positivo.")
    marker = output_root / ".prepared.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"La salida {output_root} está incompleta. Revísela antes de reintentar."
        )
    if not (raw_root / "images").exists() or not (raw_root / "labels").exists():
        raise FileNotFoundError(
            f"Faltan images/ o labels/ en {raw_root}; ejecute primero la descarga."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    split_aliases = {"train": "train", "val": "validation", "test": "test"}
    counts: Counter[tuple[str, str]] = Counter()
    skipped = Counter()
    source_images = 0
    crop_records: list[dict[str, object]] = []

    for source_split, target_split in split_aliases.items():
        image_dir = raw_root / "images" / source_split
        label_dir = raw_root / "labels" / source_split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_images += len(images)
        for image_path in tqdm(images, desc=f"LVIS {source_split}", unit="img"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                skipped["missing_label_file"] += 1
                continue
            boxes = _read_yolo_boxes(label_path)
            if not boxes:
                skipped["image_without_boxes"] += 1
                continue
            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            except OSError:
                skipped["invalid_image"] += 1
                continue

            image_width, image_height = image.size
            source_key = image_path.relative_to(raw_root).as_posix()
            digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:20]
            for box_index, (class_id, x_center, y_center, width, height) in enumerate(boxes):
                object_width = width * image_width
                object_height = height * image_height
                if min(object_width, object_height) < min_object_size:
                    skipped["object_too_small"] += 1
                    continue

                half_width = object_width * (0.5 + margin_ratio)
                half_height = object_height * (0.5 + margin_ratio)
                center_x = x_center * image_width
                center_y = y_center * image_height
                left = max(0, int(round(center_x - half_width)))
                top = max(0, int(round(center_y - half_height)))
                right = min(image_width, int(round(center_x + half_width)))
                bottom = min(image_height, int(round(center_y + half_height)))
                if right - left < min_object_size or bottom - top < min_object_size:
                    skipped["invalid_crop"] += 1
                    continue

                label = normalize_label(LVIS_CLASS_NAMES[class_id])
                destination_dir = output_root / target_split / label
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{digest}_box{box_index:03d}.jpg"
                image.crop((left, top, right, bottom)).save(
                    destination,
                    format="JPEG",
                    quality=92,
                    optimize=True,
                )
                counts[(target_split, label)] += 1
                crop_records.append(
                    {
                        "crop_path": destination.relative_to(output_root).as_posix(),
                        "source_image": source_key,
                        "source_split": source_split,
                        "target_split": target_split,
                        "box_index": box_index,
                        "source_class_id": class_id,
                        "original_label": label,
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    }
                )

    crops_manifest = output_root / "crops_manifest.csv"
    manifest_fields = [
        "crop_path",
        "source_image",
        "source_split",
        "target_split",
        "box_index",
        "source_class_id",
        "original_label",
        "left",
        "top",
        "right",
        "bottom",
    ]
    with crops_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(crop_records)

    report: dict[str, object] = {
        "source_images": source_images,
        "generated_crops": int(sum(counts.values())),
        "crops_manifest": str(crops_manifest),
        "margin_ratio": margin_ratio,
        "min_object_size": min_object_size,
        "skipped": dict(sorted(skipped.items())),
        "crops_by_split_and_label": {
            f"{split}/{label}": count
            for (split, label), count in sorted(counts.items())
        },
    }
    marker.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def prepare_registered_lvis(
    paths: ProjectPaths,
    dataset_id: str = "lvis_fruits_vegetables",
    margin_ratio: float = 0.15,
    min_object_size: int = 24,
) -> dict[str, object]:
    dataset = _dataset_entry(paths, dataset_id)
    raw_root = (paths.root / str(dataset.get("download_path", dataset["path"]))).resolve()
    output_root = (paths.root / str(dataset["path"])).resolve()
    data_root = (paths.root / "data").resolve()
    if not raw_root.is_relative_to(data_root) or not output_root.is_relative_to(data_root):
        raise ValueError("Las rutas LVIS deben permanecer dentro de data/.")
    return prepare_lvis_dataset(raw_root, output_root, margin_ratio, min_object_size)
