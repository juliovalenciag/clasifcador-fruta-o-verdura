"""Caché de características clásicas indexada por contenido (SHA-256).

Extraer descriptores es la etapa más costosa del pipeline y requiere tener las
imágenes originales en disco (~10 GB). Esta caché guarda una sola vez el vector
de cada imagen y permite:

- entrenar y comparar modelos sin volver a leer imágenes;
- compartir con el equipo un paquete ligero (vectores + manifiesto reducido)
  para entrenar sin descargar los datasets originales;
- agregar imágenes nuevas de forma incremental: solo se extraen los SHA-256
  que aún no existen en la caché.

Estructura en disco::

    data/features/<extractor_id>/
        features.npy      matriz float32 (N x D); fila i <-> index.sha256[i]
        index.csv         sha256 de cada fila
        manifest.csv      filas elegibles con fuente, etiqueta y split
        meta.json         configuración del extractor y versiones
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .classic_features import FeatureConfig, extract_classic_features


# Incrementar cuando cambie la lógica de extract_classic_features: invalida las
# cachés anteriores aunque FeatureConfig no cambie.
EXTRACTOR_VERSION = "classic-v4"

# Columnas mínimas para reconstruir splits y filtros sin las imágenes.
STORE_MANIFEST_COLUMNS = [
    "image_id",
    "source",
    "dataset_role",
    "relative_path",
    "original_label",
    "class_id",
    "mapping_status",
    "width",
    "height",
    "sha256",
    "valid",
    "duplicate_group",
    "split",
]


def extractor_id(config: FeatureConfig) -> str:
    """Identificador estable del extractor: versión + tamaño + hash de la configuración."""

    payload = json.dumps(
        {"version": EXTRACTOR_VERSION, **config.to_dict()}, sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{EXTRACTOR_VERSION}-{config.image_size}px-{digest}"


def default_store_dir(root: Path, config: FeatureConfig) -> Path:
    return root / "data" / "features" / extractor_id(config)


@dataclass
class FeatureStore:
    directory: Path
    features: np.ndarray
    sha256: np.ndarray
    manifest: pd.DataFrame
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        self._rows = {value: index for index, value in enumerate(self.sha256)}

    def __len__(self) -> int:
        return len(self.sha256)

    @property
    def feature_config(self) -> FeatureConfig:
        values = dict(self.meta["feature_config"])
        for key in ("hog_pixels_per_cell", "hog_cells_per_block"):
            values[key] = tuple(values[key])
        return FeatureConfig(**values)

    def lookup(self, hashes: pd.Series | list[str]) -> np.ndarray:
        """Devuelve la fila de cada SHA-256; -1 cuando no está en la caché."""

        return np.asarray(
            [self._rows.get(str(value), -1) for value in hashes], dtype=np.int64
        )


def load_feature_store(directory: Path) -> FeatureStore:
    directory = Path(directory)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No existe una caché de características en {directory}. Ejecute "
            "`frutas-data extract-features` o `frutas-data hub pull-features`."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = np.load(directory / "features.npy", mmap_mode="r")
    index = pd.read_csv(directory / "index.csv", dtype=str)
    manifest = pd.read_csv(
        directory / "manifest.csv", dtype=str, keep_default_na=False
    )
    if len(index) != features.shape[0]:
        raise ValueError(
            f"Caché inconsistente en {directory}: {len(index)} hashes y "
            f"{features.shape[0]} filas de características."
        )
    return FeatureStore(
        directory=directory,
        features=features,
        sha256=index["sha256"].to_numpy(),
        manifest=manifest,
        meta=meta,
    )


def _extract_for_store(task: tuple[str, str, dict[str, Any]]) -> tuple[str, np.ndarray | None, str]:
    # Función de nivel superior para que ProcessPoolExecutor pueda serializarla.
    sha, path, config_values = task
    for key in ("hog_pixels_per_cell", "hog_cells_per_block"):
        config_values[key] = tuple(config_values[key])
    try:
        vector = extract_classic_features(Path(path), FeatureConfig(**config_values))
        return sha, vector, ""
    except (OSError, ValueError) as exc:
        return sha, None, f"{path}: {exc}"


def _store_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    from .classical_training import _eligible

    parts = [
        _eligible(manifest, split)
        for split in ("train", "validation", "test", "external")
    ]
    frame = pd.concat(parts, ignore_index=True)
    for column in STORE_MANIFEST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[STORE_MANIFEST_COLUMNS]


def build_feature_store(
    manifest_path: Path,
    store_dir: Path,
    config: FeatureConfig | None = None,
    workers: int | None = None,
) -> dict[str, object]:
    """Extrae (o completa) la caché para todas las imágenes elegibles del manifiesto."""

    config = config or FeatureConfig()
    store_dir = Path(store_dir)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    eligible = _store_manifest(manifest)
    paths_by_sha = (
        manifest.loc[manifest["sha256"].isin(eligible["sha256"]), ["sha256", "path"]]
        .drop_duplicates("sha256")
        .set_index("sha256")["path"]
    )

    existing_features: np.ndarray | None = None
    existing_hashes: list[str] = []
    if (store_dir / "meta.json").exists():
        previous = load_feature_store(store_dir)
        if previous.meta.get("extractor_id") != extractor_id(config):
            raise ValueError(
                f"{store_dir} pertenece a otro extractor "
                f"({previous.meta.get('extractor_id')}); use otro directorio."
            )
        existing_features = np.asarray(previous.features)
        existing_hashes = list(previous.sha256)

    known = set(existing_hashes)
    pending = [
        (sha, str(paths_by_sha[sha]), config.to_dict())
        for sha in eligible["sha256"].drop_duplicates()
        if sha not in known and sha in paths_by_sha.index
    ]
    new_hashes: list[str] = []
    new_vectors: list[np.ndarray] = []
    errors: list[str] = []
    if pending:
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_extract_for_store, pending, chunksize=32)
            for sha, vector, error in tqdm(
                results, total=len(pending), desc="Extrayendo características", unit="img"
            ):
                if vector is None:
                    errors.append(error)
                    continue
                new_hashes.append(sha)
                new_vectors.append(vector)

    blocks = [block for block in (existing_features,) if block is not None]
    if new_vectors:
        blocks.append(np.stack(new_vectors).astype(np.float32))
    if not blocks:
        raise ValueError("No fue posible extraer características de ninguna imagen.")
    features = np.concatenate(blocks, axis=0)
    hashes = existing_hashes + new_hashes

    store_dir.mkdir(parents=True, exist_ok=True)
    # Se escribe a un temporal y luego se reemplaza para no corromper la caché.
    temporary = store_dir / "features.tmp.npy"
    np.save(temporary, features)
    temporary.replace(store_dir / "features.npy")
    pd.DataFrame({"sha256": hashes}).to_csv(store_dir / "index.csv", index=False)
    eligible[eligible["sha256"].isin(set(hashes))].to_csv(
        store_dir / "manifest.csv", index=False
    )

    import sklearn
    import skimage

    meta = {
        "extractor_id": extractor_id(config),
        "extractor_version": EXTRACTOR_VERSION,
        "feature_config": config.to_dict(),
        "feature_count": int(features.shape[1]),
        "images": int(features.shape[0]),
        "scikit_learn_version": sklearn.__version__,
        "scikit_image_version": skimage.__version__,
        "splits": eligible[eligible["sha256"].isin(set(hashes))]["split"]
        .value_counts()
        .to_dict(),
    }
    (store_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "store_dir": str(store_dir.resolve()),
        "extractor_id": meta["extractor_id"],
        "images": meta["images"],
        "new_images": len(new_hashes),
        "reused_images": len(existing_hashes),
        "errors": errors[:20],
        "error_count": len(errors),
    }
