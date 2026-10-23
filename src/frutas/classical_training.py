from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.base import BaseEstimator, clone
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from .baseline import CLASSES
from .classic_features import (
    AdaptivePCA,
    BlockBalancedScaler,
    FeatureConfig,
    extract_classic_features,
    safe_rgb,
)
from .feature_store import FeatureStore


def _eligible(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    valid_mask = (
        (frame["valid"].astype(str).str.lower() == "true")
        & (frame["mapping_status"] == "approved")
        & (frame["class_id"].isin(CLASSES))
    )
    approved = frame[valid_mask].copy()
    if (
        not approved.empty
        and {"source", "relative_path", "width", "height"}.issubset(approved.columns)
    ):
        is_lvis = approved["source"] == "lvis_fruits_vegetables"
        if is_lvis.any():
            lvis_rows = approved.loc[is_lvis]
            scene_ids = (
                lvis_rows["relative_path"]
                .astype(str)
                .str.extract(r"([0-9a-f]{20})_box\d+", expand=False)
            )
            has_scene = scene_ids.notna()
            if has_scene.any():
                widths = pd.to_numeric(lvis_rows["width"], errors="coerce").fillna(0.0)
                heights = pd.to_numeric(lvis_rows["height"], errors="coerce").fillna(
                    0.0
                )
                min_side = np.minimum(widths, heights)
                grouped = lvis_rows.assign(_scene_id=scene_ids).groupby("_scene_id")
                scene_boxes = grouped["class_id"].transform("count")
                scene_classes = grouped["class_id"].transform("nunique")
                clean_lvis = (~has_scene) | (
                    (min_side >= 64.0) & (scene_classes == 1) & (scene_boxes <= 8)
                )
                keep_index = approved.index[~is_lvis].union(
                    lvis_rows.index[clean_lvis]
                )
                approved = approved.loc[keep_index]
    return approved[approved["split"] == split].copy().reset_index(drop=True)


def _sample_per_class(
    frame: pd.DataFrame, maximum: int | None, seed: int
) -> pd.DataFrame:
    if not maximum or maximum <= 0:
        return frame
    parts = [
        group.sample(min(len(group), maximum), random_state=seed)
        for _, group in frame.groupby("class_id", sort=True)
    ]
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _sample_per_source_label(
    frame: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    """Evita que etiquetas masivas de una fuente dominen las características."""

    if not maximum or maximum <= 0 or not {"source", "original_label"}.issubset(frame):
        return frame
    parts = [
        group.sample(min(len(group), maximum), random_state=seed)
        for _, group in frame.groupby(["source", "original_label"], sort=True)
    ]
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _balanced_sample_indices(
    labels: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    """Selecciona hasta ``maximum`` filas, repartidas equitativamente entre las clases."""

    labels = np.asarray(labels)
    if not maximum or maximum <= 0 or len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    quota = maximum // len(classes)
    selected: list[int] = []
    remaining: list[int] = []
    for class_id in classes:
        positions = np.flatnonzero(labels == class_id)
        positions = rng.permutation(positions)
        take = min(quota, len(positions))
        selected.extend(positions[:take].tolist())
        remaining.extend(positions[take:].tolist())

    missing = maximum - len(selected)
    if missing > 0 and remaining:
        selected.extend(rng.permutation(remaining)[:missing].tolist())
    return np.asarray(rng.permutation(selected), dtype=np.int64)


def _fit_subset(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    svm_max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name.startswith("svm_"):
        indices = _balanced_sample_indices(labels, svm_max_rows, seed)
        return features[indices], labels[indices]
    return features, labels


def _stable_rng(path: str, seed: int, variant: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{path}|{seed}|{variant}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(value)


def _tta_views(source: Image.Image) -> list[Image.Image]:
    """Genera 4 vistas deterministas en tiempo de prueba (TTA) para mitigar encuadre e iluminación."""

    rgb = safe_rgb(source)
    width, height = rgb.size
    margin_x = max(1, int(round(width * 0.07)))
    margin_y = max(1, int(round(height * 0.07)))
    if width - 2 * margin_x >= 8 and height - 2 * margin_y >= 8:
        central = rgb.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    else:
        central = rgb
    return [
        rgb,
        ImageOps.mirror(rgb),
        central,
        ImageOps.autocontrast(rgb, cutoff=1),
    ]


def _make_noisy_variant(
    source: Image.Image,
    source_key: str,
    seed: int,
    variant: int,
    target_size: int,
    heavy: bool = False,
) -> Image.Image:
    """Crea una variación reproducible sin escribir copias en disco."""

    rng = _stable_rng(source_key, seed, variant)
    working_size = max(target_size * 2, 96)
    image = ImageOps.fit(
        source,
        (working_size, working_size),
        method=Image.Resampling.LANCZOS,
    )

    width, height = image.size
    scale = float(rng.uniform(0.78 if heavy else 0.88, 1.0))
    crop_width = max(2, int(width * scale))
    crop_height = max(2, int(height * scale))
    left = int(rng.integers(0, max(width - crop_width + 1, 1)))
    top = int(rng.integers(0, max(height - crop_height + 1, 1)))
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    max_angle = 20.0 if heavy else 12.0
    image = image.rotate(
        float(rng.uniform(-max_angle, max_angle)),
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0)),
    )
    b_low, b_high = (0.76, 1.24) if heavy else (0.90, 1.10)
    c_low, c_high = (0.80, 1.20) if heavy else (0.92, 1.08)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(b_low, b_high)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(c_low, c_high)))
    image = ImageEnhance.Color(image).enhance(float(rng.uniform(c_low, c_high)))
    if heavy and rng.random() < 0.35:
        image = image.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0)))
        )

    pixels = np.asarray(image, dtype=np.float32)
    sigma = float(rng.uniform(2.0, 9.0) if heavy else rng.uniform(0.8, 3.0))
    pixels = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(pixels, mode="RGB")


def _extract_features(
    frame: pd.DataFrame,
    config: FeatureConfig,
    augmentations: int = 0,
    seed: int = 42,
    description: str = "Extrayendo características",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    accepted_rows: list[int] = []

    def extract_row(
        item: tuple[int, pd.Series],
    ) -> tuple[int, str, list[np.ndarray], str]:
        index, row = item
        try:
            path = Path(row["path"])
            with Image.open(path) as opened:
                source = safe_rgb(opened)
            row_vectors = [extract_classic_features(source, config)]
            for variant in range(augmentations):
                noisy = _make_noisy_variant(
                    source,
                    str(path.resolve()),
                    seed,
                    variant,
                    config.image_size,
                    heavy=False,
                )
                row_vectors.append(extract_classic_features(noisy, config))
            return index, str(row["class_id"]), row_vectors, ""
        except (OSError, ValueError) as exc:
            return index, str(row["class_id"]), [], f"{row['path']}: {exc}"

    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(extract_row, frame.iterrows())
        for index, class_id, row_vectors, error in tqdm(
            results, total=len(frame), desc=description, unit="img"
        ):
            if error:
                print(f"Aviso: se omite {error}")
                continue
            vectors.extend(row_vectors)
            labels.extend([class_id] * len(row_vectors))
            accepted_rows.append(index)

    if not vectors:
        raise ValueError("No fue posible extraer características de ninguna imagen.")
    accepted = frame.loc[accepted_rows].reset_index(drop=True)
    return np.stack(vectors), np.asarray(labels), accepted


def _features_for(
    frame: pd.DataFrame,
    config: FeatureConfig,
    augmentations: int = 0,
    seed: int = 42,
    description: str = "Extrayendo características",
    store: FeatureStore | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Usa la caché por SHA-256 cuando existe y solo extrae lo que falta."""

    if store is None or augmentations > 0 or "sha256" not in frame:
        return _extract_features(frame, config, augmentations, seed, description)

    rows = store.lookup(frame["sha256"])
    cached = rows >= 0
    missing = frame.loc[~cached]
    if "path" in missing:
        missing = missing[missing["path"].astype(str).map(lambda v: bool(v) and Path(v).exists())]
    skipped = int((~cached).sum()) - len(missing)
    if skipped:
        print(
            f"Aviso: {description}: se omiten {skipped} imágenes que no están en la "
            "caché ni en disco."
        )
    parts_x = [np.asarray(store.features[rows[cached]], dtype=np.float32)]
    parts_y = [frame.loc[cached, "class_id"].astype(str).to_numpy()]
    parts_frame = [frame.loc[cached]]
    if not missing.empty:
        x_new, y_new, accepted_new = _extract_features(
            missing.reset_index(drop=True), config, 0, seed, f"{description} (sin caché)"
        )
        parts_x.append(x_new)
        parts_y.append(y_new)
        parts_frame.append(accepted_new)
    accepted = pd.concat(parts_frame, ignore_index=True)
    if accepted.empty:
        raise ValueError(f"{description}: no hay características disponibles.")
    return np.concatenate(parts_x), np.concatenate(parts_y), accepted


def _candidate_estimators(
    seed: int,
    config: FeatureConfig | None = None,
    n_samples: int = 5000,
) -> dict[str, BaseEstimator]:
    cfg = config or FeatureConfig()
    small_regime = n_samples < 100
    cv_folds = 2 if small_regime else 3
    et_trees = 64 if small_regime else 500
    et_leaf = 1 if small_regime else 3
    hgb_iter_63 = 60 if small_regime else 1000
    hgb_iter_95 = 60 if small_regime else 1000
    hgb_leaf_63 = max(1, min(20, n_samples // 5))
    hgb_leaf_95 = max(1, min(15, n_samples // 5))
    use_early_stop = not small_regime

    make_scaler = lambda: BlockBalancedScaler(
        color_dim=cfg.color_feature_count,
        lbp_dim=cfg.lbp_feature_count,
    )
    scaled_logistic = lambda c: Pipeline(
        [
            ("scale", make_scaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "logistic_c1": scaled_logistic(1.0),
        "sgd_log": Pipeline(
            [
                ("scale", make_scaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=2000,
                        tol=1e-4,
                        average=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf_c10": Pipeline(
            [
                ("scale", make_scaler()),
                ("pca", AdaptivePCA(max_components=256, random_state=seed)),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        SVC(
                            C=10.0,
                            gamma="scale",
                            class_weight="balanced",
                            cache_size=2048,
                            random_state=seed,
                        ),
                        cv=cv_folds,
                        ensemble=False,
                    ),
                ),
            ]
        ),
        "svm_rbf_c25": Pipeline(
            [
                ("scale", make_scaler()),
                ("pca", AdaptivePCA(max_components=256, random_state=seed)),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        SVC(
                            C=25.0,
                            gamma="scale",
                            class_weight="balanced",
                            cache_size=2048,
                            random_state=seed,
                        ),
                        cv=cv_folds,
                        ensemble=False,
                    ),
                ),
            ]
        ),
        "extra_trees_sqrt": ExtraTreesClassifier(
            n_estimators=et_trees,
            max_features="sqrt",
            min_samples_leaf=et_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees_040": ExtraTreesClassifier(
            n_estimators=et_trees,
            max_features=0.25,
            min_samples_leaf=et_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hgb_depth_63": HistGradientBoostingClassifier(
            max_iter=hgb_iter_63,
            learning_rate=0.08,
            max_leaf_nodes=63,
            min_samples_leaf=hgb_leaf_63,
            l2_regularization=1.5,
            class_weight="balanced",
            early_stopping=use_early_stop,
            n_iter_no_change=40,
            random_state=seed,
        ),
        "hgb_depth_95": HistGradientBoostingClassifier(
            max_iter=hgb_iter_95,
            learning_rate=0.06,
            max_leaf_nodes=95,
            min_samples_leaf=hgb_leaf_95,
            l2_regularization=2.0,
            class_weight="balanced",
            early_stopping=use_early_stop,
            n_iter_no_change=40,
            random_state=seed + 7,
        ),
    }


def _aligned_probabilities(
    estimator: BaseEstimator, features: np.ndarray
) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = [str(value) for value in estimator.classes_]
    aligned = np.zeros((len(features), len(CLASSES)), dtype=np.float64)
    for output_index, class_id in enumerate(CLASSES):
        if class_id in classes:
            aligned[:, output_index] = probabilities[:, classes.index(class_id)]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sums, np.finfo(np.float64).eps)


@dataclass
class SoftVotingEnsemble:
    estimators: list[tuple[str, BaseEstimator]]
    weights: list[float] | None = None
    classes_: np.ndarray = field(
        default_factory=lambda: np.asarray(CLASSES), init=False, repr=False
    )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrices = [
            _aligned_probabilities(model, features) for _, model in self.estimators
        ]
        return np.average(np.stack(matrices), axis=0, weights=self.weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[indices]


@dataclass
class ClassicalImageClassifier:
    feature_config: FeatureConfig
    estimator: BaseEstimator | SoftVotingEnsemble
    model_name: str
    uncertainty_threshold: float = 0.60
    decision_threshold: float = 0.50
    metadata: dict[str, Any] = field(default_factory=dict)

    def _format_prediction(
        self,
        probabilities: np.ndarray,
        tta_views: int = 1,
    ) -> dict[str, object]:
        decision_threshold = float(getattr(self, "decision_threshold", 0.50))
        index = 0 if float(probabilities[0]) >= decision_threshold else 1
        confidence = float(probabilities[index])
        return {
            "prediction": CLASSES[index],
            "confidence": confidence,
            "probabilities": {
                class_id: float(probabilities[position])
                for position, class_id in enumerate(CLASSES)
            },
            "uncertain": confidence < self.uncertainty_threshold,
            "model": self.model_name,
            "decision_threshold": decision_threshold,
            "tta_views": tta_views,
        }

    def predict_vector(self, vector: np.ndarray) -> dict[str, object]:
        probabilities = _aligned_probabilities(self.estimator, vector.reshape(1, -1))[0]
        return self._format_prediction(probabilities, tta_views=1)

    def predict_image(
        self,
        image: Image.Image,
        tta: bool = True,
    ) -> dict[str, object]:
        views = _tta_views(image) if tta else [safe_rgb(image)]
        matrix = np.stack(
            [extract_classic_features(view, self.feature_config) for view in views]
        )
        probabilities = _aligned_probabilities(self.estimator, matrix).mean(axis=0)
        return self._format_prediction(probabilities, tta_views=len(views))

    def predict_path(self, path: str | Path, tta: bool = True) -> dict[str, object]:
        target = Path(path)
        with Image.open(target) as opened:
            rgb = safe_rgb(opened)
        result = self.predict_image(rgb, tta=tta)
        result["path"] = str(target)
        return result


def _metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    decision_threshold: float = 0.50,
) -> dict[str, object]:
    predictions = np.where(
        probabilities[:, 0] >= decision_threshold, CLASSES[0], CLASSES[1]
    )
    predicted_indices = np.where(predictions == CLASSES[0], 0, 1)
    confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
    accepted = confidence >= threshold
    report = classification_report(
        truth,
        predictions,
        labels=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, object] = {
        "images": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision_macro": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(truth, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix_labels": list(CLASSES),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(CLASSES)
        ).tolist(),
        "class_report": report,
        "uncertainty_threshold": threshold,
        "decision_threshold": decision_threshold,
        "coverage": float(accepted.mean()),
        "uncertain_images": int((~accepted).sum()),
    }
    if accepted.any():
        result["accuracy_when_confident"] = float(
            accuracy_score(truth[accepted], predictions[accepted])
        )
    else:
        result["accuracy_when_confident"] = None
    return result


def _select_decision_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    uncertainty_threshold: float = 0.60,
) -> tuple[float, dict[str, object]]:
    """Ajusta el corte binario usando validation, nunca test."""

    best_threshold = 0.50
    best_metrics = _metrics(
        truth, probabilities, uncertainty_threshold, best_threshold
    )
    best_key = (
        float(best_metrics["f1_macro"]),
        float(best_metrics["balanced_accuracy"]),
        float(best_metrics["accuracy"]),
        -abs(best_threshold - 0.50),
    )
    for candidate in np.linspace(0.30, 0.70, 81):
        candidate = round(float(candidate), 3)
        metrics = _metrics(truth, probabilities, uncertainty_threshold, candidate)
        key = (
            float(metrics["f1_macro"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["accuracy"]),
            -abs(candidate - 0.50),
        )
        if key > best_key:
            best_threshold = candidate
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def _selection_row(
    name: str,
    metrics: dict[str, object],
    seconds: float,
    fit_rows: int,
) -> dict[str, object]:
    return {
        "model": name,
        "f1_macro": metrics["f1_macro"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "decision_threshold": metrics["decision_threshold"],
        "fit_seconds": round(seconds, 3),
        "fit_rows": fit_rows,
    }


def train_classical_suite(
    manifest_path: Path,
    model_path: Path,
    metrics_path: Path,
    selection_path: Path,
    feature_config: FeatureConfig | None = None,
    max_per_class: int | None = None,
    max_per_source_label: int | None = 0,
    svm_max_rows: int | None = 14000,
    augmentations: int = 0,
    uncertainty_threshold: float = 0.60,
    seed: int = 42,
    feature_store: FeatureStore | None = None,
    eval_max_per_source_label: int | None = 180,
    excluded_train_sha256: set[str] | None = None,
) -> tuple[ClassicalImageClassifier, dict[str, object]]:
    """Selecciona en validación, reentrena y toca test una sola vez.

    ``max_per_source_label`` limita solo train (0 = todas las imágenes; usar todo
    train sube ~2.5 pts de F1 en validation frente al tope histórico de 180). Validation y test
    conservan ``eval_max_per_source_label`` para que ninguna fuente masiva domine
    la selección ni las métricas reportadas. ``excluded_train_sha256`` (revisión
    manual de ``audit-labels``) solo se descarta de train.
    """

    if augmentations < 0:
        raise ValueError("augmentations no puede ser negativo.")
    if svm_max_rows is not None and svm_max_rows < 0:
        raise ValueError("svm_max_rows no puede ser negativo.")
    if not 0.5 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold debe estar entre 0.5 y 1.0.")

    config = feature_config or FeatureConfig()
    if feature_store is not None:
        if feature_store.feature_config != config:
            raise ValueError(
                "La caché de características se generó con otra FeatureConfig: "
                f"{feature_store.meta.get('extractor_id')}."
            )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    eligible_train = _eligible(manifest, "train")
    excluded_count = 0
    if excluded_train_sha256 and "sha256" in eligible_train:
        excluded_mask = eligible_train["sha256"].isin(excluded_train_sha256)
        excluded_count = int(excluded_mask.sum())
        eligible_train = eligible_train[~excluded_mask].reset_index(drop=True)
    train = _sample_per_source_label(
        _sample_per_class(eligible_train, max_per_class, seed),
        max_per_source_label,
        seed,
    )
    validation = _sample_per_source_label(
        _sample_per_class(_eligible(manifest, "validation"), max_per_class, seed + 1),
        eval_max_per_source_label,
        seed + 1,
    )
    test = _sample_per_source_label(
        _sample_per_class(_eligible(manifest, "test"), max_per_class, seed + 2),
        eval_max_per_source_label,
        seed + 2,
    )
    external = _sample_per_class(
        _eligible(manifest, "external"), max_per_class, seed + 3
    )
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.empty:
            raise ValueError(f"El split obligatorio {name!r} está vacío.")

    x_train, y_train, accepted_train = _features_for(
        train, config, augmentations, seed, "Train", feature_store
    )
    x_validation, y_validation, accepted_validation = _features_for(
        validation, config, 0, seed, "Validación", feature_store
    )
    x_test, y_test, accepted_test = _features_for(
        test, config, 0, seed, "Test", feature_store
    )
    if set(np.unique(y_train)) != set(CLASSES):
        raise ValueError("Train debe contener ejemplos válidos de fruta y verdura.")

    templates = _candidate_estimators(seed, config, n_samples=len(x_train))
    fitted: dict[str, BaseEstimator] = {}
    validation_metrics: dict[str, dict[str, object]] = {}
    decision_thresholds: dict[str, float] = {}
    selection_rows: list[dict[str, object]] = []
    for name, estimator in templates.items():
        print(f"Entrenando candidato: {name}")
        fit_x, fit_y = _fit_subset(name, x_train, y_train, svm_max_rows, seed)
        if len(fit_y) != len(y_train):
            print(
                f"  {name} usa {len(fit_y)} filas balanceadas de {len(y_train)} "
                "para evitar sesgo de clase mayoritaria y mantener viable el ajuste."
            )
        started = perf_counter()
        estimator.fit(fit_x, fit_y)
        elapsed = perf_counter() - started
        probabilities = _aligned_probabilities(estimator, x_validation)
        decision_threshold, candidate_metrics = _select_decision_threshold(
            y_validation, probabilities, uncertainty_threshold
        )
        fitted[name] = estimator
        validation_metrics[name] = candidate_metrics
        decision_thresholds[name] = decision_threshold
        selection_rows.append(
            _selection_row(name, candidate_metrics, elapsed, len(fit_y))
        )

    ranked_names = sorted(
        fitted,
        key=lambda name: (
            float(validation_metrics[name]["f1_macro"]),
            float(validation_metrics[name]["balanced_accuracy"]),
            float(validation_metrics[name]["accuracy"]),
        ),
        reverse=True,
    )
    top_names = ranked_names[: min(4, len(ranked_names))]
    voting_weights = [
        max(float(validation_metrics[name]["f1_macro"]) - 0.45, 0.05) ** 2
        for name in top_names
    ]
    voting = SoftVotingEnsemble(
        [(name, fitted[name]) for name in top_names],
        weights=voting_weights,
    )
    voting_threshold, voting_metrics = _select_decision_threshold(
        y_validation, voting.predict_proba(x_validation), uncertainty_threshold
    )
    validation_metrics["soft_voting_top3"] = voting_metrics
    decision_thresholds["soft_voting_top3"] = voting_threshold
    selection_rows.append(_selection_row("soft_voting_top3", voting_metrics, 0.0, 0))

    selected_name = max(
        validation_metrics,
        key=lambda name: (
            float(validation_metrics[name]["f1_macro"]),
            float(validation_metrics[name]["balanced_accuracy"]),
            float(validation_metrics[name]["accuracy"]),
        ),
    )
    print(f"Seleccionado con validation: {selected_name}")

    x_final = np.concatenate((x_train, x_validation), axis=0)
    y_final = np.concatenate((y_train, y_validation), axis=0)
    final_templates = _candidate_estimators(seed, config, n_samples=len(x_final))
    if selected_name == "soft_voting_top3":
        final_members: list[tuple[str, BaseEstimator]] = []
        for name in top_names:
            fit_x, fit_y = _fit_subset(
                name, x_final, y_final, svm_max_rows, seed + 1
            )
            member = clone(final_templates[name]).fit(fit_x, fit_y)
            final_members.append((name, member))
        final_estimator: BaseEstimator | SoftVotingEnsemble = SoftVotingEnsemble(
            final_members,
            weights=voting_weights,
        )
    else:
        fit_x, fit_y = _fit_subset(
            selected_name, x_final, y_final, svm_max_rows, seed + 1
        )
        final_estimator = clone(final_templates[selected_name]).fit(fit_x, fit_y)

    artifact = ClassicalImageClassifier(
        feature_config=config,
        estimator=final_estimator,
        model_name=selected_name,
        uncertainty_threshold=uncertainty_threshold,
        decision_threshold=decision_thresholds[selected_name],
        metadata={
            "seed": seed,
            "augmentations_per_train_image": augmentations,
            "max_per_source_label": max_per_source_label,
            "eval_max_per_source_label": eval_max_per_source_label,
            "excluded_train_images": excluded_count,
            "svm_max_rows": svm_max_rows,
            "feature_store": feature_store.meta.get("extractor_id")
            if feature_store is not None
            else None,
            "selected_on": "validation",
            "decision_threshold_selected_on": "validation",
            "scikit_learn_version": sklearn_version,
            "top_ensemble_members": top_names,
            "top_ensemble_weights": [round(float(w), 4) for w in voting_weights],
        },
    )
    test_metrics = _metrics(
        y_test,
        _aligned_probabilities(final_estimator, x_test),
        uncertainty_threshold,
        decision_thresholds[selected_name],
    )

    external_metrics: dict[str, object] | None = None
    accepted_external = external.iloc[0:0].copy()
    if not external.empty:
        x_external, y_external, accepted_external = _features_for(
            external, config, 0, seed, "Evaluación externa", feature_store
        )
        external_metrics = _metrics(
            y_external,
            _aligned_probabilities(final_estimator, x_external),
            uncertainty_threshold,
            decision_thresholds[selected_name],
        )

    selection = pd.DataFrame(selection_rows).sort_values(
        ["f1_macro", "balanced_accuracy", "accuracy"], ascending=False
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)

    metrics: dict[str, object] = {
        "selected_model": selected_name,
        "selection_rule": "mayor F1 macro; desempate por balanced accuracy y accuracy",
        "decision_threshold": decision_thresholds[selected_name],
        "feature_config": config.to_dict(),
        "feature_count": int(x_train.shape[1]),
        "augmentations_per_train_image": augmentations,
        "max_per_source_label": max_per_source_label,
        "eval_max_per_source_label": eval_max_per_source_label,
        "excluded_train_images": excluded_count,
        "svm_max_rows": svm_max_rows,
        "original_images": {
            "train": len(accepted_train),
            "validation": len(accepted_validation),
            "test": len(accepted_test),
            "external": len(accepted_external),
        },
        "fit_rows_after_augmentation": int(len(x_final)),
        "validation": validation_metrics,
        "test": test_metrics,
        "external": external_metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return artifact, metrics


def load_classical_model(path: Path) -> ClassicalImageClassifier:
    model = joblib.load(path)
    if not isinstance(model, ClassicalImageClassifier):
        raise TypeError(
            f"El artefacto {path} no contiene un ClassicalImageClassifier."
        )
    return model


def _load_known_label_lookup(model_path: Path) -> dict[str, str]:
    from .manifest import normalize_label

    lookup: dict[str, str] = {class_id: class_id for class_id in CLASSES}
    candidates = [
        model_path.resolve().parent.parent / "configs" / "label_mapping.csv",
        Path.cwd().resolve() / "configs" / "label_mapping.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            frame = pd.read_csv(candidate, dtype=str).fillna("")
            for record in frame.to_dict("records"):
                if (
                    record.get("status") == "approved"
                    and record.get("class_id") in CLASSES
                ):
                    lookup[normalize_label(record["original_label"])] = record[
                        "class_id"
                    ]
            break
    return lookup


def _infer_expected_class(
    image_path: Path,
    root: Path,
    lookup: dict[str, str] | None = None,
) -> str:
    from .manifest import normalize_label

    mapping = lookup or {class_id: class_id for class_id in CLASSES}
    try:
        relative_parts = image_path.relative_to(root).parts[:-1]
    except ValueError:
        relative_parts = image_path.parts[:-1]
    for part in reversed(relative_parts):
        resolved = mapping.get(normalize_label(part), "")
        if resolved in CLASSES:
            return resolved
    return mapping.get(normalize_label(image_path.parent.name), "")


def predict_target(
    target_path: Path,
    model_path: Path,
    output_path: Path | None = None,
    tta: bool = True,
) -> dict[str, object]:
    """Predice una sola imagen o un directorio completo de fotografías reales."""

    from .manifest import IMAGE_EXTENSIONS

    model = load_classical_model(model_path)
    target = target_path.resolve()
    if not target.exists():
        raise FileNotFoundError(f"No existe la ruta a clasificar: {target}")
    lookup = _load_known_label_lookup(model_path)

    if target.is_file():
        result = model.predict_path(target, tta=tta)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "path": str(target),
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "prob_fruta": result["probabilities"]["fruta"],
                        "prob_verdura": result["probabilities"]["verdura"],
                        "uncertain": result["uncertain"],
                        "model": result["model"],
                        "tta_views": result["tta_views"],
                    }
                ]
            ).to_csv(output_path, index=False)
            result["output_path"] = str(output_path.resolve())
        return result

    images = [
        path
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rows: list[dict[str, object]] = []
    probabilities_list: list[list[float]] = []
    truth_list: list[str] = []
    errors: list[str] = []

    for image_path in images:
        try:
            pred = model.predict_path(image_path, tta=tta)
        except (OSError, ValueError) as exc:
            errors.append(f"{image_path}: {exc}")
            continue
        expected = _infer_expected_class(image_path, target, lookup)
        probs = pred["probabilities"]
        row: dict[str, object] = {
            "relative_path": image_path.relative_to(target).as_posix(),
            "path": str(image_path),
            "prediction": pred["prediction"],
            "confidence": round(float(pred["confidence"]), 4),
            "prob_fruta": round(float(probs["fruta"]), 4),
            "prob_verdura": round(float(probs["verdura"]), 4),
            "uncertain": bool(pred["uncertain"]),
            "expected_class": expected,
            "correct": (pred["prediction"] == expected) if expected else None,
        }
        rows.append(row)
        if expected in CLASSES:
            truth_list.append(expected)
            probabilities_list.append([float(probs["fruta"]), float(probs["verdura"])])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    counts = {
        class_id: sum(1 for item in rows if item["prediction"] == class_id)
        for class_id in CLASSES
    }
    summary: dict[str, object] = {
        "mode": "directory",
        "target_path": str(target),
        "model": model.model_name,
        "tta": bool(tta),
        "total_images": len(rows),
        "predicted_counts": counts,
        "uncertain_images": sum(1 for item in rows if item["uncertain"]),
        "average_confidence": (
            float(np.mean([float(item["confidence"]) for item in rows]))
            if rows
            else 0.0
        ),
        "skipped_errors": errors,
    }
    if truth_list and len(truth_list) == len(rows):
        summary["labeled_metrics"] = _metrics(
            np.asarray(truth_list),
            np.asarray(probabilities_list, dtype=np.float64),
            model.uncertainty_threshold,
            float(getattr(model, "decision_threshold", 0.50)),
        )
    if output_path is not None:
        summary["output_path"] = str(output_path.resolve())
    summary["predictions"] = rows
    return summary


NOISE_CONDITIONS = (
    "clean",
    "gaussian_noise",
    "gaussian_blur",
    "illumination_low",
    "illumination_high",
    "rotation_crop",
    "occlusion_patch",
    "combined_heavy",
)


def _apply_noise_condition(
    source: Image.Image,
    condition: str,
    source_key: str,
    seed: int,
    target_size: int,
) -> Image.Image:
    """Aplica una perturbación controlada para evaluar robustez ante ruido."""

    rgb = safe_rgb(source)
    working_size = max(target_size * 2, 96)
    image = ImageOps.fit(
        rgb,
        (working_size, working_size),
        method=Image.Resampling.LANCZOS,
    )
    rng = _stable_rng(f"{source_key}:{condition}", seed, 0)

    if condition == "clean":
        return image
    if condition == "gaussian_noise":
        pixels = np.asarray(image, dtype=np.float32)
        noisy = np.clip(pixels + rng.normal(0.0, 14.0, pixels.shape), 0, 255).astype(
            np.uint8
        )
        return Image.fromarray(noisy, mode="RGB")
    if condition == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    if condition == "illumination_low":
        dark = ImageEnhance.Brightness(image).enhance(0.62)
        return ImageEnhance.Contrast(dark).enhance(0.85)
    if condition == "illumination_high":
        bright = ImageEnhance.Brightness(image).enhance(1.38)
        return ImageEnhance.Contrast(bright).enhance(0.85)
    if condition == "rotation_crop":
        fill = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
        rotated = image.rotate(
            18.0,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=fill,
        )
        width, height = rotated.size
        crop_w = max(8, int(width * 0.82))
        crop_h = max(8, int(height * 0.82))
        left = max(0, (width - crop_w) // 3)
        top = max(0, (height - crop_h) // 3)
        return rotated.crop((left, top, left + crop_w, top + crop_h)).resize(
            (width, height), Image.Resampling.LANCZOS
        )
    if condition == "occlusion_patch":
        pixels = np.asarray(image, dtype=np.uint8).copy()
        height, width, _ = pixels.shape
        patch_h = max(4, int(height * 0.22))
        patch_w = max(4, int(width * 0.22))
        top = int(
            rng.integers(
                height // 5, max(height // 5 + 1, height - patch_h - height // 5)
            )
        )
        left = int(
            rng.integers(width // 5, max(width // 5 + 1, width - patch_w - width // 5))
        )
        neutral = pixels.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        pixels[top : top + patch_h, left : left + patch_w] = neutral
        return Image.fromarray(pixels, mode="RGB")
    if condition == "combined_heavy":
        return _make_noisy_variant(rgb, source_key, seed, 0, target_size, heavy=True)
    raise ValueError(f"Condición de ruido desconocida: {condition}")


def evaluate_noise_robustness(
    manifest_path: Path,
    model_path: Path,
    split: str = "test",
    output_path: Path | None = None,
    max_per_source_label: int | None = 50,
    tta: bool = True,
    seed: int = 42,
) -> dict[str, object]:
    """Evalúa la caída y retención de desempeño bajo 8 condiciones de ruido explícitas."""

    if split not in {"validation", "test", "external"}:
        raise ValueError("split debe ser validation, test o external.")
    model = load_classical_model(model_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    evaluation = _sample_per_source_label(
        _eligible(manifest, split),
        max_per_source_label,
        seed,
    )
    if evaluation.empty:
        return {"split": split, "images": 0, "conditions": []}

    opened_images: list[tuple[Image.Image, str, str]] = []
    for _, row in evaluation.iterrows():
        try:
            path = Path(row["path"])
            with Image.open(path) as img:
                opened_images.append(
                    (safe_rgb(img), str(path.resolve()), str(row["class_id"]))
                )
        except (OSError, ValueError):
            continue

    if not opened_images:
        return {"split": split, "images": 0, "conditions": []}

    truth = np.asarray([item[2] for item in opened_images])
    condition_rows: list[dict[str, object]] = []
    clean_f1: float | None = None
    workers = min(8, os.cpu_count() or 4)

    for condition in NOISE_CONDITIONS:

        def _extract_one(item: tuple[Image.Image, str, str]) -> np.ndarray:
            rgb, key, _ = item
            perturbed = _apply_noise_condition(
                rgb,
                condition,
                key,
                seed,
                model.feature_config.image_size,
            )
            views = _tta_views(perturbed) if tta else [perturbed]
            return np.stack(
                [extract_classic_features(view, model.feature_config) for view in views]
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            view_stacks = list(
                tqdm(
                    executor.map(_extract_one, opened_images),
                    total=len(opened_images),
                    desc=f"Ruido [{condition}]",
                    unit="img",
                )
            )
        n_samples = len(view_stacks)
        n_views = view_stacks[0].shape[0]
        flat_features = np.concatenate(view_stacks, axis=0)
        flat_probs = _aligned_probabilities(model.estimator, flat_features)
        probabilities = flat_probs.reshape(n_samples, n_views, len(CLASSES)).mean(
            axis=1
        )
        cond_metrics = _metrics(
            truth,
            probabilities,
            model.uncertainty_threshold,
            float(getattr(model, "decision_threshold", 0.50)),
        )
        f1_val = float(cond_metrics["f1_macro"])
        if condition == "clean":
            clean_f1 = f1_val
        retention = (
            100.0 * f1_val / clean_f1 if clean_f1 and clean_f1 > 0 else 100.0
        )
        condition_rows.append(
            {
                "condition": condition,
                "images": int(cond_metrics["images"]),
                "accuracy": round(float(cond_metrics["accuracy"]), 4),
                "balanced_accuracy": round(
                    float(cond_metrics["balanced_accuracy"]), 4
                ),
                "f1_macro": round(f1_val, 4),
                "accuracy_when_confident": (
                    round(float(cond_metrics["accuracy_when_confident"]), 4)
                    if cond_metrics["accuracy_when_confident"] is not None
                    else None
                ),
                "coverage": round(float(cond_metrics["coverage"]), 4),
                "f1_retention_pct": round(retention, 2),
            }
        )

    table = pd.DataFrame(condition_rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)

    non_clean = [row for row in condition_rows if row["condition"] != "clean"]
    return {
        "split": split,
        "model": model.model_name,
        "tta": bool(tta),
        "images": len(opened_images),
        "clean_f1_macro": clean_f1,
        "mean_noisy_f1_macro": (
            round(float(np.mean([float(r["f1_macro"]) for r in non_clean])), 4)
            if non_clean
            else clean_f1
        ),
        "mean_f1_retention_pct": (
            round(
                float(np.mean([float(r["f1_retention_pct"]) for r in non_clean])), 2
            )
            if non_clean
            else 100.0
        ),
        "conditions": condition_rows,
    }


def evaluate_classical_model(
    manifest_path: Path,
    model_path: Path,
    split: str = "test",
    breakdown_path: Path | None = None,
    excluded_sources: tuple[str, ...] = (),
    tta: bool = True,
) -> dict[str, object]:
    """Evalúa un artefacto y desglosa errores por fuente y etiqueta original."""

    if split not in {"validation", "test", "external"}:
        raise ValueError("split debe ser validation, test o external.")
    model = load_classical_model(model_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    evaluation = _eligible(manifest, split)
    if excluded_sources and "source" in evaluation:
        evaluation = evaluation[
            ~evaluation["source"].isin(excluded_sources)
        ].reset_index(drop=True)
    if evaluation.empty:
        return {
            "split": split,
            "images": 0,
            "tta": bool(tta),
            "excluded_sources": list(excluded_sources),
            "breakdown": [],
        }

    if tta:
        view_features: list[np.ndarray] = []
        truth_rows: list[str] = []
        accepted_indices: list[int] = []

        def _extract_views(
            item: tuple[int, pd.Series],
        ) -> tuple[int, str, np.ndarray | None]:
            index, row = item
            try:
                with Image.open(Path(row["path"])) as opened:
                    views = _tta_views(opened)
                    matrix = np.stack(
                        [
                            extract_classic_features(view, model.feature_config)
                            for view in views
                        ]
                    )
                return index, str(row["class_id"]), matrix
            except (OSError, ValueError):
                return index, str(row["class_id"]), None

        workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_extract_views, evaluation.iterrows())
            for index, class_id, matrix in tqdm(
                results,
                total=len(evaluation),
                desc=f"Evaluando {split} con TTA",
                unit="img",
            ):
                if matrix is None:
                    continue
                view_features.append(matrix)
                truth_rows.append(class_id)
                accepted_indices.append(index)
        if not view_features:
            raise ValueError(
                "No fue posible extraer características para la evaluación."
            )
        stacked = np.stack(view_features)
        flat = stacked.reshape(-1, stacked.shape[-1])
        flat_probabilities = _aligned_probabilities(model.estimator, flat)
        probabilities = flat_probabilities.reshape(
            len(stacked), stacked.shape[1], len(CLASSES)
        ).mean(axis=1)
        truth = np.asarray(truth_rows)
        accepted = evaluation.loc[accepted_indices].reset_index(drop=True)
    else:
        features, truth, accepted = _extract_features(
            evaluation,
            model.feature_config,
            description=f"Evaluando {split}",
        )
        probabilities = _aligned_probabilities(model.estimator, features)
    decision_threshold = float(getattr(model, "decision_threshold", 0.50))
    predictions = np.where(
        probabilities[:, 0] >= decision_threshold, CLASSES[0], CLASSES[1]
    )
    predicted_indices = np.where(predictions == CLASSES[0], 0, 1)
    confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
    accepted = accepted.copy()
    accepted["prediction"] = predictions
    accepted["confidence"] = confidence
    accepted["correct"] = predictions == truth
    accepted["uncertain"] = confidence < model.uncertainty_threshold
    if "source" not in accepted:
        accepted["source"] = "unknown"
    if "original_label" not in accepted:
        accepted["original_label"] = accepted["class_id"]

    breakdown = (
        accepted.groupby(["source", "class_id", "original_label"], as_index=False)
        .agg(
            images=("correct", "size"),
            accuracy=("correct", "mean"),
            average_confidence=("confidence", "mean"),
            uncertain_rate=("uncertain", "mean"),
        )
        .sort_values(["accuracy", "images"], ascending=[True, False])
    )
    if breakdown_path is not None:
        breakdown_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown.to_csv(breakdown_path, index=False)

    result = _metrics(
        truth,
        probabilities,
        model.uncertainty_threshold,
        decision_threshold,
    )
    result.update(
        split=split,
        tta=bool(tta),
        excluded_sources=list(excluded_sources),
        breakdown=breakdown.to_dict("records"),
    )
    return result
