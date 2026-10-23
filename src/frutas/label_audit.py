"""Auditoría de etiquetas sospechosas en train mediante predicciones out-of-fold.

Las fuentes web (sobre todo ``fruits_youssef``) incluyen platos, jugos, ensaladas
y escenas mixtas que la guía del equipo ordena excluir. Revisarlas todas a mano es
inviable; este módulo prioriza la revisión:

1. Entrena un Gradient Boosting rápido con validación cruzada sobre train (desde la
   caché de características) y obtiene, para cada imagen, la probabilidad que un
   modelo que *no la vio* asigna a su propia etiqueta.
2. Las imágenes con probabilidad baja se exportan a un CSV (y, si hay imágenes en
   disco, a hojas de contacto) para revisión humana.
3. Las filas que una persona marca con ``excluir = si`` se agregan a
   ``configs/excluded_images.csv``, que ``train-classic`` descarta **solo de train**.

Validation y test no se filtran con esta lista: la selección de candidatos depende
del propio modelo y limpiar las particiones de evaluación así inflaría las métricas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import CLASSES
from .feature_store import FeatureStore


EXCLUSION_COLUMNS = ["sha256", "source", "original_label", "class_id", "reason"]
CONFIRM_VALUES = {"si", "sí", "s", "x", "1", "true", "excluir"}


def load_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return set(frame["sha256"]) - {""}


def audit_train_labels(
    store: FeatureStore,
    manifest: pd.DataFrame,
    output_csv: Path,
    folds: int = 3,
    probability_threshold: float = 0.15,
    seed: int = 42,
    already_excluded: set[str] | None = None,
) -> dict[str, object]:
    """Exporta las imágenes de train cuya etiqueta contradice un modelo out-of-fold."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from .classical_training import _eligible

    train = _eligible(manifest, "train")
    train = train[~train["sha256"].isin(already_excluded or set())]
    train = train.drop_duplicates("sha256").reset_index(drop=True)
    rows = store.lookup(train["sha256"])
    train = train[rows >= 0].reset_index(drop=True)
    rows = rows[rows >= 0]
    if train["class_id"].nunique() < 2:
        raise ValueError("Train debe contener fruta y verdura para auditar etiquetas.")

    features = np.asarray(store.features[rows], dtype=np.float32)
    labels = train["class_id"].to_numpy()
    estimator = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.5,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        estimator, features, labels, cv=splitter, method="predict_proba"
    )
    # cross_val_predict ordena columnas como np.unique(labels) == CLASSES.
    label_index = np.searchsorted(np.asarray(CLASSES), labels)
    given = probabilities[np.arange(len(labels)), label_index]

    audited = train.assign(p_given_label=np.round(given, 4))
    suspects = (
        audited[audited["p_given_label"] < probability_threshold]
        .sort_values("p_given_label")
        .reset_index(drop=True)
    )
    columns = [
        "sha256",
        "source",
        "original_label",
        "class_id",
        "p_given_label",
        "relative_path",
    ]
    if "path" in suspects:
        columns.append("path")
    report = suspects[columns].assign(excluir="", reason="")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)

    by_label = (
        audited.assign(suspect=audited["p_given_label"] < probability_threshold)
        .groupby(["source", "original_label"])["suspect"]
        .agg(["sum", "mean"])
        .sort_values("sum", ascending=False)
    )
    return {
        "train_images": int(len(audited)),
        "suspects": int(len(report)),
        "suspect_rate": float(len(report) / max(len(audited), 1)),
        "probability_threshold": probability_threshold,
        "output_csv": str(output_csv.resolve()),
        "top_labels": [
            {
                "source": source,
                "original_label": label,
                "suspects": int(values["sum"]),
                "rate": round(float(values["mean"]), 4),
            }
            for (source, label), values in by_label.head(15).iterrows()
        ],
    }


def write_contact_sheets(
    suspects_csv: Path,
    output_dir: Path,
    per_sheet: int = 48,
    thumb: int = 128,
) -> list[Path]:
    """Genera mosaicos numerados (fila del CSV) para revisar rápido las sospechas."""

    from PIL import Image, ImageDraw

    from .classic_features import safe_rgb

    frame = pd.read_csv(suspects_csv, dtype=str, keep_default_na=False)
    if "path" not in frame:
        raise ValueError("El CSV no tiene rutas de imagen; se requiere el dataset local.")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    columns = 8
    for start in range(0, len(frame), per_sheet):
        chunk = frame.iloc[start : start + per_sheet]
        rows = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, (index, record) in enumerate(chunk.iterrows()):
            x, y = (offset % columns) * thumb, (offset // columns) * (thumb + 14)
            try:
                with Image.open(record["path"]) as opened:
                    image = safe_rgb(opened)
                image.thumbnail((thumb, thumb))
                sheet.paste(image, (x, y))
            except OSError:
                pass
            draw.text(
                (x + 2, y + thumb),
                f"{index} {record['class_id'][:3]} {record['original_label'][:12]}",
                fill="black",
            )
        path = output_dir / f"sospechosas_{start:05d}.jpg"
        sheet.save(path, quality=85)
        written.append(path)
    return written


def apply_label_review(reviewed_csv: Path, exclusions_path: Path) -> dict[str, object]:
    """Agrega a la lista de exclusión las filas confirmadas con ``excluir = si``."""

    reviewed = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    confirmed = reviewed[
        reviewed["excluir"].str.strip().str.lower().isin(CONFIRM_VALUES)
    ].copy()
    confirmed["reason"] = confirmed["reason"].replace("", "revisión manual")
    current = (
        pd.read_csv(exclusions_path, dtype=str, keep_default_na=False)
        if exclusions_path.exists()
        else pd.DataFrame(columns=EXCLUSION_COLUMNS)
    )
    merged = (
        pd.concat([current, confirmed[EXCLUSION_COLUMNS]], ignore_index=True)
        .drop_duplicates("sha256", keep="first")
        .sort_values(["source", "original_label", "sha256"])
    )
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    merged[EXCLUSION_COLUMNS].to_csv(exclusions_path, index=False)
    return {
        "confirmed_in_review": int(len(confirmed)),
        "total_excluded": int(len(merged)),
        "exclusions_path": str(exclusions_path.resolve()),
    }
