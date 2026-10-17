from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .config import ProjectPaths, load_dataset_config


def _dataset_entry(paths: ProjectPaths, dataset_id: str) -> dict[str, object]:
    for dataset in load_dataset_config(paths.datasets_config):
        if dataset.get("id") == dataset_id:
            return dataset
    raise ValueError(f"Dataset desconocido: '{dataset_id}'.")


def _download_git_paths(
    repo_url: str,
    revision: str | None,
    relative_paths: list[str] | None,
    destination: Path,
    skip_lfs: bool = False,
) -> None:
    """Clona un repositorio Git en un directorio temporal y copia las rutas solicitadas."""

    with TemporaryDirectory(prefix="frutas-git-") as temporary:
        checkout = Path(temporary) / "repo"
        environment = os.environ.copy()
        if skip_lfs:
            environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        clone_cmd = ["git", "clone", "--depth", "1"]
        if relative_paths:
            clone_cmd.extend(["--filter=blob:none", "--sparse"])
        clone_cmd.extend([repo_url, str(checkout)])
        subprocess.run(
            clone_cmd,
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
        )
        if relative_paths:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "sparse-checkout",
                    "set",
                    # El modo cone normalmente acepta solo directorios. Algunos
                    # datasets también requieren archivos raíz como data.yaml y
                    # README.md; --skip-checks permite combinarlos con carpetas.
                    "--skip-checks",
                    *relative_paths,
                ],
                check=True,
                env=environment,
                stdout=subprocess.DEVNULL,
            )
        if revision:
            current = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if current != revision:
                fetch_cmd = ["git", "-C", str(checkout), "fetch", "--depth", "1"]
                if relative_paths:
                    fetch_cmd.append("--filter=blob:none")
                fetch_cmd.extend(["origin", revision])
                subprocess.run(
                    fetch_cmd,
                    check=True,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                )
                subprocess.run(
                    ["git", "-C", str(checkout), "checkout", "--detach", revision],
                    check=True,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                )

        paths_to_copy = relative_paths or ["."]
        for relative in paths_to_copy:
            source = (checkout / relative).resolve()
            if not source.exists():
                raise RuntimeError(
                    f"La ruta '{relative}' no existe en {repo_url}."
                )
            target = destination if relative in {".", ""} else destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git"),
                )
            else:
                shutil.copy2(source, target)


def _download_huggingface_git_metadata(
    repo_id: str,
    revision: str | None,
    relative_paths: list[str],
    destination: Path,
) -> None:
    """Obtiene muchos archivos pequeños en un solo paquete Git.

    Hugging Face puede limitar miles de solicitudes HTTP individuales. Un clon
    con Git LFS desactivado trae las etiquetas de texto de forma eficiente y
    deja los archivos pesados como punteros; estos últimos se descargan aparte
    mediante ``snapshot_download``.
    """

    repo_url = f"https://huggingface.co/datasets/{repo_id}.git"
    _download_git_paths(
        repo_url=repo_url,
        revision=revision,
        relative_paths=relative_paths,
        destination=destination,
        skip_lfs=True,
    )


def download_dataset(
    paths: ProjectPaths,
    dataset_id: str,
    force: bool = False,
) -> Path:
    """Descarga una fuente registrada en la ruta reproducible configurada."""

    dataset = _dataset_entry(paths, dataset_id)
    kaggle_handle = str(dataset.get("kaggle_handle", "")).strip()
    huggingface_repo = str(dataset.get("huggingface_repo", "")).strip()
    git_repo = str(dataset.get("git_repo", "")).strip()
    if not kaggle_handle and not huggingface_repo and not git_repo:
        raise ValueError(
            f"El dataset '{dataset_id}' no tiene una fuente descargable configurada; "
            "debe obtenerse manualmente desde la URL registrada."
        )

    configured_destination = dataset.get("download_path", dataset["path"])
    destination = (paths.root / str(configured_destination)).resolve()
    allowed_root = (paths.root / "data").resolve()
    if not destination.is_relative_to(allowed_root):
        raise ValueError(f"Ruta de descarga fuera de data/: {destination}")

    completion_marker = destination / ".complete"
    if completion_marker.exists() and not force:
        print(f"Dataset ya disponible en {destination}")
        return destination
    resumable_huggingface = bool(
        huggingface_repo and (destination / ".cache" / "huggingface").exists()
    )
    if (
        destination.exists()
        and any(destination.iterdir())
        and not force
        and not resumable_huggingface
    ):
        raise RuntimeError(
            f"La carpeta {destination} ya contiene archivos pero no está marcada como "
            "descarga completa. Revísela y use --force únicamente si desea reemplazarla."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if kaggle_handle:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "Falta kagglehub. Instale el proyecto con: "
                "pip install -e '.[dev,notebooks]'"
            ) from exc
        downloaded = kagglehub.dataset_download(
            kaggle_handle,
            output_dir=str(destination),
            force_download=force,
        )
        source_reference = kaggle_handle
    elif huggingface_repo:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Falta huggingface_hub. Instale el proyecto con: "
                "pip install -e '.[dev,notebooks]'"
            ) from exc
        revision = str(dataset.get("huggingface_revision", "")).strip() or None
        allow_patterns = dataset.get("allow_patterns") or None
        downloaded = snapshot_download(
            repo_id=huggingface_repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            allow_patterns=allow_patterns,
            max_workers=int(dataset.get("download_workers", 32)),
            force_download=force,
        )
        git_metadata_paths = [
            str(item) for item in dataset.get("huggingface_git_metadata_paths", [])
        ]
        if git_metadata_paths:
            _download_huggingface_git_metadata(
                huggingface_repo,
                revision,
                git_metadata_paths,
                destination,
            )
        source_reference = (
            f"{huggingface_repo}@{revision}" if revision else huggingface_repo
        )
    else:
        revision = str(dataset.get("git_revision", "")).strip() or None
        include_paths = [
            str(item) for item in dataset.get("git_include_paths", [])
        ] or None
        if force and destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        _download_git_paths(
            repo_url=git_repo,
            revision=revision,
            relative_paths=include_paths,
            destination=destination,
            skip_lfs=False,
        )
        downloaded = str(destination)
        source_reference = f"{git_repo}@{revision}" if revision else git_repo
    result = Path(downloaded).resolve()
    if not result.exists():
        raise RuntimeError(
            f"La fuente informó una descarga en {result}, pero la ruta no existe."
        )
    # KaggleHub puede crear `.complete` como directorio de metadatos. Si no lo
    # hace, dejamos nuestro propio marcador de texto con la misma semántica.
    if not completion_marker.exists():
        completion_marker.write_text(
            json.dumps(
                {"dataset_id": dataset_id, "source_reference": source_reference},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Dataset '{dataset_id}' disponible en {result}")
    return result
