"""Sincronización de modelos y cachés de características con Hugging Face Hub.

El equipo comparte dos repositorios (configurados en ``configs/hub.yaml``):

- un repositorio de **modelo** con ``classical_model.joblib`` y sus métricas;
- un repositorio de **dataset** con las cachés de características
  (``features/<extractor_id>/``), suficientes para entrenar sin imágenes.

Cada ``push`` crea un commit en el Hub; ``pull`` puede fijar una revisión
(commit o etiqueta) para que todos usen exactamente el mismo artefacto.

La autenticación usa ``HF_TOKEN`` o ``hf auth login``; el token nunca se guarda
en el repositorio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import ProjectPaths


MODEL_FILES = (
    "classical_model.joblib",
    "classical_metrics.json",
    "model_selection.csv",
    "error_analysis_by_label.csv",
)


def load_hub_config(paths: ProjectPaths) -> dict[str, Any]:
    config_path = paths.root / "configs" / "hub.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No existe {config_path}.")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key in ("model_repo", "features_repo"):
        if not str(config.get(key, "")).strip():
            raise ValueError(f"configs/hub.yaml debe definir {key!r}.")
    return config


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _ensure_repo(repo_id: str, repo_type: str, private: bool) -> None:
    _api().create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)


def _model_card(metrics: dict[str, Any], repo_id: str) -> str:
    test = metrics.get("test") or {}
    lines = [
        "---",
        "library_name: scikit-learn",
        "tags: [image-classification, classical-ml, fruta-verdura]",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Clasificador binario clásico (sin redes neuronales) `fruta` / `verdura` "
        "bajo criterio botánico. Generado por `frutas-data hub push-model`.",
        "",
        "| Campo | Valor |",
        "|---|---|",
        f"| Modelo seleccionado | `{metrics.get('selected_model', '?')}` |",
        f"| Umbral de decisión | `{metrics.get('decision_threshold', '?')}` |",
        f"| Características | `{metrics.get('feature_count', '?')}` |",
    ]
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "accuracy_when_confident"):
        if test.get(key) is not None:
            lines.append(f"| Test `{key}` | `{float(test[key]):.4f}` |")
    lines += [
        "",
        "Uso:",
        "",
        "```bash",
        "frutas-data hub pull-model",
        "frutas-data predict-classic mi_foto.jpg",
        "```",
        "",
        "> El artefacto es un `joblib` (pickle): cárguelo solo desde este repositorio "
        "privado del equipo y con la misma versión de scikit-learn.",
        "",
    ]
    return "\n".join(lines)


def push_model(
    paths: ProjectPaths,
    artifacts_dir: Path,
    message: str | None = None,
    tag: str | None = None,
) -> dict[str, object]:
    """Sube el modelo vigente y sus reportes al repositorio de modelo del equipo."""

    from huggingface_hub import CommitOperationAdd

    config = load_hub_config(paths)
    repo_id = str(config["model_repo"])
    model_path = artifacts_dir / MODEL_FILES[0]
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}; ejecute train-classic.")
    metrics_path = artifacts_dir / "classical_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(artifacts_dir / name))
        for name in MODEL_FILES
        if (artifacts_dir / name).exists()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=_model_card(metrics, repo_id).encode("utf-8"),
        )
    )
    _ensure_repo(repo_id, "model", bool(config.get("private", True)))
    test = metrics.get("test") or {}
    default_message = (
        f"Modelo {metrics.get('selected_model', '')} · "
        f"test F1 macro {float(test.get('f1_macro', 0.0)):.4f}"
    )
    commit = _api().create_commit(
        repo_id,
        operations,
        commit_message=message or default_message,
        repo_type="model",
    )
    if tag:
        _api().create_tag(repo_id, tag=tag, revision=commit.oid, repo_type="model")
    return {
        "repo": repo_id,
        "commit": commit.oid,
        "url": commit.commit_url,
        "tag": tag,
        "files": [operation.path_in_repo for operation in operations],
    }


def pull_model(
    paths: ProjectPaths,
    artifacts_dir: Path,
    revision: str | None = None,
) -> dict[str, object]:
    """Descarga el modelo compartido en ``artifacts/`` sin necesidad de entrenar."""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    config = load_hub_config(paths)
    repo_id = str(config["model_repo"])
    revision = revision or config.get("model_revision") or None
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for name in MODEL_FILES:
        try:
            local = hf_hub_download(
                repo_id,
                name,
                repo_type="model",
                revision=revision,
                local_dir=artifacts_dir,
            )
        except EntryNotFoundError:
            if name == MODEL_FILES[0]:
                raise
            continue
        downloaded.append(str(Path(local).resolve()))
    return {"repo": repo_id, "revision": revision or "main", "files": downloaded}


def push_features(
    paths: ProjectPaths,
    store_dir: Path,
    message: str | None = None,
) -> dict[str, object]:
    """Sube una caché de características (vectores + manifiesto reducido)."""

    config = load_hub_config(paths)
    repo_id = str(config["features_repo"])
    meta_path = store_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No existe {meta_path}; ejecute primero `frutas-data extract-features`."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    store_id = str(meta["extractor_id"])
    _ensure_repo(repo_id, "dataset", bool(config.get("private", True)))
    commit = _api().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(store_dir),
        path_in_repo=f"features/{store_id}",
        allow_patterns=["features.npy", "index.csv", "manifest.csv", "meta.json"],
        commit_message=message
        or f"Caché {store_id}: {meta.get('images', '?')} imágenes",
    )
    return {
        "repo": repo_id,
        "extractor_id": store_id,
        "images": meta.get("images"),
        "commit": commit.oid,
        "url": commit.commit_url,
    }


def list_remote_feature_stores(paths: ProjectPaths, revision: str | None = None) -> list[str]:
    config = load_hub_config(paths)
    files = _api().list_repo_files(
        str(config["features_repo"]), repo_type="dataset", revision=revision
    )
    return sorted(
        {
            Path(name).parts[1]
            for name in files
            if name.startswith("features/") and name.endswith("meta.json")
        }
    )


def pull_features(
    paths: ProjectPaths,
    store_id: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    """Descarga una caché a ``data/features/<extractor_id>/`` (cientos de MB, no 10 GB)."""

    from huggingface_hub import snapshot_download

    config = load_hub_config(paths)
    repo_id = str(config["features_repo"])
    if store_id is None:
        from .classic_features import FeatureConfig
        from .feature_store import extractor_id

        store_id = extractor_id(FeatureConfig())
    available = list_remote_feature_stores(paths, revision)
    if store_id not in available:
        raise ValueError(
            f"La caché {store_id!r} no existe en {repo_id}. Disponibles: {available}"
        )
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"features/{store_id}/*"],
        local_dir=paths.root / "data",
    )
    store_dir = paths.root / "data" / "features" / store_id
    return {"repo": repo_id, "extractor_id": store_id, "store_dir": str(store_dir)}
