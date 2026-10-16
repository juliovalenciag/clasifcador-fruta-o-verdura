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
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from .baseline import CLASSES
from .classic_features import FeatureConfig, extract_classic_features


def _eligible(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    mask = (
        (frame["split"] == split)
        & (frame["valid"].astype(str).str.lower() == "true")
        & (frame["mapping_status"] == "approved")
        & (frame["class_id"].isin(CLASSES))
    )
    return frame[mask].copy().reset_index(drop=True)


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
    """Selecciona hasta ``maximum`` filas, repartidas entre las clases.

    SVC con kernel RBF crece de forma aproximadamente cuadrática con el número de
    filas. El resto de los modelos puede aprovechar el conjunto completo, mientras
    SVC recibe una muestra reproducible y balanceada.
    """

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
    if not model_name.startswith("svm_"):
        return features, labels
    indices = _balanced_sample_indices(labels, svm_max_rows, seed)
    return features[indices], labels[indices]


def _stable_rng(path: str, seed: int, variant: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{path}|{seed}|{variant}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(value)


def _make_noisy_variant(
    source: Image.Image,
    source_key: str,
    seed: int,
    variant: int,
    target_size: int,
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
    scale = float(rng.uniform(0.78, 1.0))
    crop_width = max(2, int(width * scale))
    crop_height = max(2, int(height * scale))
    left = int(rng.integers(0, max(width - crop_width + 1, 1)))
    top = int(rng.integers(0, max(height - crop_height + 1, 1)))
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    image = image.rotate(
        float(rng.uniform(-22.0, 22.0)),
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0)),
    )
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.72, 1.28)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.72, 1.28)))
    image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.78, 1.22)))
    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.4))))

    pixels = np.asarray(image, dtype=np.float32)
    sigma = float(rng.uniform(2.0, 12.0))
    pixels = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0, 255).astype(np.uint8)
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

    def extract_row(item: tuple[int, pd.Series]) -> tuple[int, str, list[np.ndarray], str]:
        index, row = item
        try:
            path = Path(row["path"])
            with Image.open(path) as opened:
                source = ImageOps.exif_transpose(opened).convert("RGB")
            row_vectors = [extract_classic_features(source, config)]
            for variant in range(augmentations):
                noisy = _make_noisy_variant(
                    source,
                    str(path.resolve()),
                    seed,
                    variant,
                    config.image_size,
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


def _candidate_estimators(seed: int) -> dict[str, BaseEstimator]:
    scaled_logistic = lambda c: Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=2500,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "logistic_c1": scaled_logistic(1.0),
        "logistic_c10": scaled_logistic(10.0),
        "sgd_log": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        max_iter=2500,
                        tol=1e-4,
                        average=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf_c10": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        cache_size=2048,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "svm_rbf_c30": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=30.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        cache_size=2048,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees_sqrt": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees_040": ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.40,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _aligned_probabilities(estimator: BaseEstimator, features: np.ndarray) -> np.ndarray:
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
        matrices = [_aligned_probabilities(model, features) for _, model in self.estimators]
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_vector(self, vector: np.ndarray) -> dict[str, object]:
        probabilities = _aligned_probabilities(self.estimator, vector.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
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
        }

    def predict_path(self, path: str | Path) -> dict[str, object]:
        vector = extract_classic_features(path, self.feature_config)
        return self.predict_vector(vector)


def _metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
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
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix_labels": list(CLASSES),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=list(CLASSES)
        ).tolist(),
        "class_report": report,
        "uncertainty_threshold": threshold,
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
    max_per_source_label: int | None = 100,
    svm_max_rows: int | None = 6000,
    augmentations: int = 1,
    uncertainty_threshold: float = 0.60,
    seed: int = 42,
) -> tuple[ClassicalImageClassifier, dict[str, object]]:
    """Selecciona en validación, reentrena y toca test una sola vez."""

    if augmentations < 0:
        raise ValueError("augmentations no puede ser negativo.")
    if svm_max_rows is not None and svm_max_rows < 0:
        raise ValueError("svm_max_rows no puede ser negativo.")
    if not 0.5 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold debe estar entre 0.5 y 1.0.")

    config = feature_config or FeatureConfig()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    train = _sample_per_source_label(
        _sample_per_class(_eligible(manifest, "train"), max_per_class, seed),
        max_per_source_label,
        seed,
    )
    validation = _sample_per_source_label(
        _sample_per_class(_eligible(manifest, "validation"), max_per_class, seed + 1),
        max_per_source_label,
        seed + 1,
    )
    test = _sample_per_source_label(
        _sample_per_class(_eligible(manifest, "test"), max_per_class, seed + 2),
        max_per_source_label,
        seed + 2,
    )
    external = _sample_per_class(
        _eligible(manifest, "external"), max_per_class, seed + 3
    )
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.empty:
            raise ValueError(f"El split obligatorio {name!r} está vacío.")

    x_train, y_train, accepted_train = _extract_features(
        train, config, augmentations, seed, "Train + ruido"
    )
    x_validation, y_validation, accepted_validation = _extract_features(
        validation, config, 0, seed, "Validación"
    )
    x_test, y_test, accepted_test = _extract_features(test, config, 0, seed, "Test"
    )
    if set(np.unique(y_train)) != set(CLASSES):
        raise ValueError("Train debe contener ejemplos válidos de fruta y verdura.")

    templates = _candidate_estimators(seed)
    fitted: dict[str, BaseEstimator] = {}
    validation_metrics: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    for name, estimator in templates.items():
        print(f"Entrenando candidato: {name}")
        fit_x, fit_y = _fit_subset(name, x_train, y_train, svm_max_rows, seed)
        if len(fit_y) != len(y_train):
            print(
                f"  SVC usa {len(fit_y)} filas balanceadas de {len(y_train)} "
                "para mantener viable el entrenamiento."
            )
        started = perf_counter()
        estimator.fit(fit_x, fit_y)
        elapsed = perf_counter() - started
        probabilities = _aligned_probabilities(estimator, x_validation)
        candidate_metrics = _metrics(
            y_validation, probabilities, uncertainty_threshold
        )
        fitted[name] = estimator
        validation_metrics[name] = candidate_metrics
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
    top_names = ranked_names[: min(3, len(ranked_names))]
    voting = SoftVotingEnsemble([(name, fitted[name]) for name in top_names])
    voting_metrics = _metrics(
        y_validation,
        voting.predict_proba(x_validation),
        uncertainty_threshold,
    )
    validation_metrics["soft_voting_top3"] = voting_metrics
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
    if selected_name == "soft_voting_top3":
        final_members: list[tuple[str, BaseEstimator]] = []
        for name in top_names:
            fit_x, fit_y = _fit_subset(
                name, x_final, y_final, svm_max_rows, seed + 1
            )
            member = clone(templates[name]).fit(fit_x, fit_y)
            final_members.append((name, member))
        final_estimator: BaseEstimator | SoftVotingEnsemble = SoftVotingEnsemble(
            final_members
        )
    else:
        fit_x, fit_y = _fit_subset(
            selected_name, x_final, y_final, svm_max_rows, seed + 1
        )
        final_estimator = clone(templates[selected_name]).fit(fit_x, fit_y)

    artifact = ClassicalImageClassifier(
        feature_config=config,
        estimator=final_estimator,
        model_name=selected_name,
        uncertainty_threshold=uncertainty_threshold,
        metadata={
            "seed": seed,
            "augmentations_per_train_image": augmentations,
            "max_per_source_label": max_per_source_label,
            "svm_max_rows": svm_max_rows,
            "selected_on": "validation",
            "top_ensemble_members": top_names,
        },
    )
    test_metrics = _metrics(
        y_test,
        _aligned_probabilities(final_estimator, x_test),
        uncertainty_threshold,
    )

    external_metrics: dict[str, object] | None = None
    accepted_external = external.iloc[0:0].copy()
    if not external.empty:
        x_external, y_external, accepted_external = _extract_features(
            external, config, 0, seed, "Evaluación externa"
        )
        external_metrics = _metrics(
            y_external,
            _aligned_probabilities(final_estimator, x_external),
            uncertainty_threshold,
        )

    selection = pd.DataFrame(selection_rows).sort_values(
        ["f1_macro", "balanced_accuracy", "accuracy"], ascending=False
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False)

    metrics: dict[str, object] = {
        "selected_model": selected_name,
        "selection_rule": "mayor F1 macro; desempate por balanced accuracy y accuracy",
        "feature_config": config.to_dict(),
        "feature_count": int(x_train.shape[1]),
        "augmentations_per_train_image": augmentations,
        "max_per_source_label": max_per_source_label,
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
        raise TypeError(f"El artefacto {path} no contiene un ClassicalImageClassifier.")
    return model


def evaluate_classical_model(
    manifest_path: Path,
    model_path: Path,
    split: str = "test",
    breakdown_path: Path | None = None,
) -> dict[str, object]:
    """Evalúa un artefacto y desglosa errores por fuente y etiqueta original."""

    if split not in {"validation", "test", "external"}:
        raise ValueError("split debe ser validation, test o external.")
    model = load_classical_model(model_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    evaluation = _eligible(manifest, split)
    if evaluation.empty:
        return {"split": split, "images": 0, "breakdown": []}

    features, truth, accepted = _extract_features(
        evaluation,
        model.feature_config,
        description=f"Evaluando {split}",
    )
    probabilities = _aligned_probabilities(model.estimator, features)
    predictions = np.asarray(CLASSES)[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
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

    result = _metrics(truth, probabilities, model.uncertainty_threshold)
    result.update(
        split=split,
        breakdown=breakdown.to_dict("records"),
    )
    return result
