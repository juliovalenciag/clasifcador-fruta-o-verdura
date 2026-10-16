from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .classic_features import FeatureConfig, extract_classic_features


CLASSES = ("fruta", "verdura")


def _fit_classifier(features: np.ndarray, labels: np.ndarray, seed: int) -> BaseEstimator:
    unique = set(np.unique(labels))
    if unique != set(CLASSES):
        raise ValueError(
            "El entrenamiento requiere ejemplos aprobados de ambas clases: fruta y verdura."
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-4,
                    random_state=seed,
                    average=True,
                ),
            ),
        ]
    ).fit(features, labels)


@dataclass
class BinaryBaseline:
    feature_config: FeatureConfig
    classifier: BaseEstimator
    uncertainty_threshold: float = 0.60

    def predict_vector(self, vector: np.ndarray) -> dict[str, object]:
        probabilities = self.classifier.predict_proba(vector.reshape(1, -1))[0]
        classes = [str(value) for value in self.classifier.classes_]
        by_class = {
            class_id: float(probability)
            for class_id, probability in zip(classes, probabilities)
        }
        index = int(np.argmax(probabilities))
        prediction = classes[index]
        confidence = float(probabilities[index])
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                class_id: by_class.get(class_id, 0.0) for class_id in CLASSES
            },
            "uncertain": confidence < self.uncertainty_threshold,
        }

    def predict_path(self, path: str | Path) -> dict[str, object]:
        return self.predict_vector(extract_classic_features(path, self.feature_config))


def _sample_per_class(
    frame: pd.DataFrame, maximum: int | None, seed: int
) -> pd.DataFrame:
    if not maximum or maximum <= 0:
        return frame
    samples = [
        group.sample(min(len(group), maximum), random_state=seed)
        for _, group in frame.groupby("class_id", sort=True)
    ]
    return (
        pd.concat(samples, ignore_index=True)
        if samples
        else frame.iloc[0:0].copy()
    )


def _extract_frame_features(
    frame: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    accepted: list[int] = []
    for index, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc="Extrayendo características",
        unit="img",
    ):
        try:
            vectors.append(extract_classic_features(row["path"], config))
            accepted.append(index)
        except (OSError, ValueError) as exc:
            print(f"Aviso: se omite {row['path']}: {exc}")
    if not vectors:
        raise ValueError("No fue posible extraer características de ninguna imagen.")
    return np.stack(vectors), frame.loc[accepted].reset_index(drop=True)


def _eligible(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    mask = (
        (frame["split"] == split)
        & (frame["valid"].astype(str).str.lower() == "true")
        & (frame["mapping_status"] == "approved")
        & (frame["class_id"].isin(CLASSES))
    )
    return frame[mask].copy()


def train_binary_baseline(
    manifest_path: Path,
    model_path: Path,
    metrics_path: Path,
    feature_config: FeatureConfig | None = None,
    max_per_class: int | None = 1000,
    seed: int = 42,
) -> tuple[BinaryBaseline, dict[str, object]]:
    feature_config = feature_config or FeatureConfig()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    train = _sample_per_class(_eligible(manifest, "train"), max_per_class, seed)
    if train.empty:
        raise ValueError("No hay datos de entrenamiento aprobados en el manifiesto.")

    train_features, train = _extract_frame_features(train, feature_config)
    classifier = _fit_classifier(
        train_features, train["class_id"].to_numpy(), seed
    )
    baseline = BinaryBaseline(feature_config=feature_config, classifier=classifier)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, model_path)

    metrics = evaluate_binary_baseline(
        baseline, manifest, "test", max_per_class, seed
    )
    metrics.update(
        train_images=len(train),
        train_class_distribution=train["class_id"]
        .value_counts()
        .sort_index()
        .to_dict(),
        feature_count=int(train_features.shape[1]),
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return baseline, metrics


def evaluate_binary_baseline(
    model: BinaryBaseline,
    manifest: pd.DataFrame,
    split: str,
    max_per_class: int | None,
    seed: int,
) -> dict[str, object]:
    evaluation = _sample_per_class(
        _eligible(manifest, split), max_per_class, seed + 1
    )
    if evaluation.empty:
        return {"evaluation_images": 0}

    features, evaluation = _extract_frame_features(
        evaluation, model.feature_config
    )
    predictions = [
        str(model.predict_vector(vector)["prediction"]) for vector in features
    ]
    truth = evaluation["class_id"].tolist()
    matrix = confusion_matrix(truth, predictions, labels=list(CLASSES))
    return {
        "evaluation_images": len(evaluation),
        "test_class_distribution": evaluation["class_id"]
        .value_counts()
        .sort_index()
        .to_dict(),
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
        "confusion_matrix": matrix.tolist(),
    }


def load_baseline(path: Path) -> BinaryBaseline:
    model = joblib.load(path)
    if not isinstance(model, BinaryBaseline):
        raise TypeError(f"El artefacto {path} no contiene un BinaryBaseline.")
    return model
