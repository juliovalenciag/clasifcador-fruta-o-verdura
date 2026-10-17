from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .baseline import CLASSES
from .classic_features import safe_rgb
from .classical_training import ClassicalImageClassifier


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class SavedTrainingExample:
    path: Path
    created: bool
    sha256: str


def decode_uploaded_image(contents: bytes) -> Image.Image:
    """Valida y decodifica una imagen subida sin conservar el archivo temporal."""

    if not contents:
        raise ValueError("El archivo está vacío.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("La imagen excede el límite de 20 MB.")
    try:
        with Image.open(BytesIO(contents)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("La imagen excede el límite de 40 megapíxeles.")
            opened.load()
            if opened.width < 16 or opened.height < 16:
                raise ValueError(
                    f"La imagen es demasiado pequeña: {opened.width}x{opened.height}."
                )
            return safe_rgb(opened).copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("El archivo no contiene una imagen válida.") from exc


def classify_uploaded_image(
    contents: bytes,
    model: ClassicalImageClassifier,
    *,
    tta: bool = True,
) -> tuple[Image.Image, dict[str, object]]:
    """Clasifica bytes de una imagen y devuelve también la vista RGB validada."""

    image = decode_uploaded_image(contents)
    return image, model.predict_image(image, tta=tta)


def save_training_example(
    contents: bytes,
    class_id: str,
    training_root: Path,
) -> SavedTrainingExample:
    """Guarda feedback confirmado en una carpeta binaria deduplicada por contenido."""

    if class_id not in CLASSES:
        raise ValueError(f"La etiqueta debe ser una de: {', '.join(CLASSES)}.")
    image = decode_uploaded_image(contents)
    canonical = BytesIO()
    image.save(canonical, format="PNG", optimize=True)
    digest = hashlib.sha256(canonical.getvalue()).hexdigest()

    destination_dir = training_root.resolve() / class_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"ui_{digest[:24]}.jpg"
    if destination.exists():
        return SavedTrainingExample(destination, False, digest)

    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=95, optimize=True)
    destination.write_bytes(encoded.getvalue())
    return SavedTrainingExample(destination, True, digest)
