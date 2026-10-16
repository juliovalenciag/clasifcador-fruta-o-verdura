from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import yaml

import frutas.download as download_module
from frutas.config import ProjectPaths
from frutas.baseline import BinaryBaseline, train_binary_baseline
from frutas.classic_features import FeatureConfig, extract_classic_features
from frutas.classical_training import (
    _balanced_sample_indices,
    evaluate_classical_model,
    load_classical_model,
    train_classical_suite,
)
from frutas.dedupe import mark_duplicates
from frutas.download import download_dataset
from frutas.manifest import build_manifest
from frutas.lvis import prepare_lvis_dataset
from frutas.splits import assign_splits


def _row(image_id: str, sha: str, phash: str, class_id: str, role: str = "trainable") -> dict[str, str]:
    return {
        "image_id": image_id,
        "source": "synthetic",
        "dataset_role": role,
        "path": f"/{image_id}.png",
        "relative_path": f"{class_id}/{image_id}.png",
        "original_label": class_id,
        "normalized_original_label": class_id,
        "class_id": class_id,
        "mapping_status": "approved",
        "original_split": "unspecified",
        "width": "32",
        "height": "32",
        "mode": "RGB",
        "format": "PNG",
        "file_size": "100",
        "sha256": sha,
        "phash": phash,
        "valid": "True",
        "error": "",
    }


def test_duplicate_groups_never_cross_splits(tmp_path: Path) -> None:
    rows = []
    for class_index, class_id in enumerate(("fruta", "verdura")):
        for index in range(12):
            image_id = f"{class_id}_{index:02d}"
            phash = f"{class_index * 1000 + index:016x}"
            rows.append(_row(image_id, f"sha_{image_id}", phash, class_id))
    rows.append(_row("fruit_copy", "sha_fruta_00", "0000000000000000", "fruta"))

    manifest = tmp_path / "manifest.csv"
    deduplicated = tmp_path / "deduplicated.csv"
    duplicates = tmp_path / "duplicates.csv"
    final = tmp_path / "final.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    marked = mark_duplicates(manifest, deduplicated, duplicates, phash_distance=0)
    assert marked.loc[marked["image_id"].isin(["fruta_00", "fruit_copy"]), "duplicate_group"].nunique() == 1

    split = assign_splits(deduplicated, final, random_state=7)
    duplicate_rows = split[split["image_id"].isin(["fruta_00", "fruit_copy"])]
    assert duplicate_rows["split"].nunique() == 1


def test_external_images_are_reserved(tmp_path: Path) -> None:
    rows = []
    for class_id in ("fruta", "verdura"):
        for index in range(10):
            image_id = f"{class_id}_{index}"
            offset = 100 if class_id == "verdura" else 0
            rows.append(_row(image_id, image_id, f"{index + offset:016x}", class_id))
    rows.append(_row("external_fruit", "external", "000000000000ffff", "fruta", role="external"))
    frame = pd.DataFrame(rows)
    frame["duplicate_group"] = frame["image_id"].map(lambda value: f"dup_{value}")
    frame["is_duplicate_candidate"] = False
    frame["duplicate_label_conflict"] = False
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    frame.to_csv(source, index=False)

    result = assign_splits(source, output, random_state=3)
    assert result.loc[result["image_id"] == "external_fruit", "split"].item() == "external"


def test_existing_splits_are_preserved_when_a_source_is_added(tmp_path: Path) -> None:
    rows = []
    for class_id in ("fruta", "verdura"):
        for index in range(12):
            row = _row(f"old_{class_id}_{index}", f"old_{class_id}_{index}", f"{index:016x}", class_id)
            row["duplicate_group"] = f"dup_old_{class_id}_{index}"
            row["is_duplicate_candidate"] = False
            row["duplicate_label_conflict"] = False
            rows.append(row)
    source = tmp_path / "source.csv"
    output = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    first = assign_splits(source, output, random_state=17)
    old_assignments = dict(zip(first["image_id"], first["split"]))

    new_rows = list(rows)
    for class_id in ("fruta", "verdura"):
        for index in range(9):
            row = _row(f"new_{class_id}_{index}", f"new_{class_id}_{index}", f"{100 + index:016x}", class_id)
            row["source"] = "second_source"
            row["duplicate_group"] = f"dup_new_{class_id}_{index}"
            row["is_duplicate_candidate"] = False
            row["duplicate_label_conflict"] = False
            new_rows.append(row)
    pd.DataFrame(new_rows).to_csv(source, index=False)

    second = assign_splits(source, output, random_state=17)
    preserved = second[second["image_id"].isin(old_assignments)]
    assert dict(zip(preserved["image_id"], preserved["split"])) == old_assignments
    assert set(preserved["split_assignment"]) == {"preserved"}


def test_published_splits_are_respected(tmp_path: Path) -> None:
    rows = []
    for class_id in ("fruta", "verdura"):
        for split_index, split_name in enumerate(("train", "validation", "test")):
            row = _row(
                f"{class_id}_{split_name}",
                f"sha_{class_id}_{split_name}",
                f"{split_index + (10 if class_id == 'verdura' else 0):016x}",
                class_id,
            )
            row["original_split"] = split_name
            row["duplicate_group"] = f"group_{class_id}_{split_name}"
            row["is_duplicate_candidate"] = False
            row["duplicate_label_conflict"] = False
            rows.append(row)

    source = tmp_path / "source.csv"
    output = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    result = assign_splits(source, output, preserve_existing=False)
    assert result["split"].tolist() == result["original_split"].tolist()
    assert set(result["split_assignment"]) == {"original"}


def test_duplicate_binary_label_conflicts_are_excluded(tmp_path: Path) -> None:
    rows = [
        _row("same_fruit", "same_sha", "00000000000000aa", "fruta"),
        _row("same_vegetable", "same_sha", "00000000000000aa", "verdura"),
    ]
    manifest = tmp_path / "manifest.csv"
    deduplicated = tmp_path / "deduplicated.csv"
    duplicates = tmp_path / "duplicates.csv"
    final = tmp_path / "final.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    marked = mark_duplicates(manifest, deduplicated, duplicates, phash_distance=0)
    assert marked["duplicate_label_conflict"].all()

    split = assign_splits(deduplicated, final)
    assert set(split["split"]) == {"excluded"}
    assert set(split["exclusion_reason"]) == {"duplicate_label_conflict"}


def test_inventory_validates_and_maps_images(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    image_dir = tmp_path / "data" / "raw" / "sample" / "train" / "Red Apple"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(image_dir / "valid.png")
    (image_dir / "broken.jpg").write_text("not an image", encoding="utf-8")

    datasets = {
        "version": 1,
        "datasets": [
            {
                "id": "sample",
                "path": "data/raw/sample",
                "role": "trainable",
                "enabled": True,
                "label_from": "parent",
                "split_aliases": {"train": ["train"]},
            }
        ],
    }
    (tmp_path / "configs" / "datasets.yaml").write_text(
        yaml.safe_dump(datasets), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "class_id": "fruta",
                "display_name_es": "Fruta",
            },
            {
                "class_id": "verdura",
                "display_name_es": "Verdura",
            },
        ]
    ).to_csv(tmp_path / "configs" / "classes.csv", index=False)
    pd.DataFrame(
        [
            {
                "source": "sample",
                "original_label": "Red Apple",
                "class_id": "fruta",
                "status": "approved",
                "notes": "",
            }
        ]
    ).to_csv(tmp_path / "configs" / "label_mapping.csv", index=False)

    result = build_manifest(ProjectPaths.from_root(tmp_path))
    assert len(result) == 2
    assert result["valid"].sum() == 1
    assert set(result["class_id"]) == {"fruta"}
    assert set(result["original_split"]) == {"train"}


def test_kaggle_download_uses_configured_destination(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "configs").mkdir()
    datasets = {
        "version": 1,
        "datasets": [
            {
                "id": "sample",
                "path": "data/raw/sample",
                "kaggle_handle": "owner/sample",
            }
        ],
    }
    (tmp_path / "configs" / "datasets.yaml").write_text(
        yaml.safe_dump(datasets), encoding="utf-8"
    )
    calls = []

    def fake_download(handle, *, output_dir, force_download):
        calls.append((handle, output_dir, force_download))
        destination = Path(output_dir)
        destination.mkdir(parents=True)
        (destination / "sample.png").write_text("downloaded", encoding="utf-8")
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=fake_download),
    )

    paths = ProjectPaths.from_root(tmp_path)
    destination = download_dataset(paths, "sample")
    assert destination == (tmp_path / "data" / "raw" / "sample").resolve()
    assert calls == [("owner/sample", str(destination), False)]
    assert (destination / ".complete").exists()

    assert download_dataset(paths, "sample") == destination
    assert len(calls) == 1


def test_huggingface_download_uses_pinned_revision(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "configs").mkdir()
    datasets = {
        "version": 1,
        "datasets": [
            {
                "id": "lvis",
                "path": "data/derived/lvis",
                "download_path": "data/raw/lvis",
                "huggingface_repo": "owner/lvis",
                "huggingface_revision": "abc123",
                "allow_patterns": ["images/**", "labels/**"],
                "huggingface_git_metadata_paths": ["labels", "data.yaml"],
            }
        ],
    }
    (tmp_path / "configs" / "datasets.yaml").write_text(
        yaml.safe_dump(datasets), encoding="utf-8"
    )
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True)
        (destination / "images").mkdir()
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    metadata_calls = []
    monkeypatch.setattr(
        download_module,
        "_download_huggingface_git_metadata",
        lambda repo, revision, paths, destination: metadata_calls.append(
            (repo, revision, paths, destination)
        ),
    )
    destination = download_dataset(ProjectPaths.from_root(tmp_path), "lvis")

    assert destination == (tmp_path / "data" / "raw" / "lvis").resolve()
    assert calls[0]["repo_id"] == "owner/lvis"
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["allow_patterns"] == ["images/**", "labels/**"]
    assert metadata_calls == [
        ("owner/lvis", "abc123", ["labels", "data.yaml"], destination)
    ]
    assert (destination / ".complete").exists()


def test_lvis_preparation_crops_yolo_objects(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "derived"
    for split in ("train", "val", "test"):
        (raw / "images" / split).mkdir(parents=True)
        (raw / "labels" / split).mkdir(parents=True)

    Image.new("RGB", (100, 80), "red").save(raw / "images" / "train" / "one.jpg")
    (raw / "labels" / "train" / "one.txt").write_text(
        "1 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )

    report = prepare_lvis_dataset(raw, output, margin_ratio=0.1, min_object_size=10)

    crops = list((output / "train" / "apple").glob("*.jpg"))
    assert report["source_images"] == 1
    assert report["generated_crops"] == 1
    assert (output / "crops_manifest.csv").exists()
    assert len(crops) == 1
    with Image.open(crops[0]) as crop:
        assert crop.width > 50
        assert crop.height > 40


def test_classic_features_are_deterministic() -> None:
    image = Image.new("RGB", (80, 50), (210, 30, 20))
    config = FeatureConfig(image_size=64)
    first = extract_classic_features(image, config)
    second = extract_classic_features(image, config)
    assert first.ndim == 1
    assert len(first) == 2068
    assert first.dtype.name == "float32"
    assert (first == second).all()


def test_block_balanced_scaler_equalizes_block_variances() -> None:
    import numpy as np
    from frutas.classic_features import BlockBalancedScaler, safe_rgb

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 2068)).astype(np.float32)
    scaler = BlockBalancedScaler(color_dim=252, lbp_dim=52).fit(x)
    z = scaler.transform(x)
    color_var = float(z[:, :252].var(axis=0).sum())
    hog_var = float(z[:, 252:2016].var(axis=0).sum())
    lbp_var = float(z[:, 2016:].var(axis=0).sum())
    assert color_var == __import__("pytest").approx(1.0, rel=1e-4)
    assert hog_var == __import__("pytest").approx(1.0, rel=1e-4)
    assert lbp_var == __import__("pytest").approx(1.0, rel=1e-4)

    palette_img = Image.new("P", (32, 32), 0)
    palette_img.info["transparency"] = b"\x00"
    rgb_img = safe_rgb(palette_img)
    assert rgb_img.mode == "RGB"



def test_svm_sampling_is_reproducible_and_balanced() -> None:
    import numpy as np

    labels = np.asarray(["fruta"] * 90 + ["verdura"] * 30)
    first = _balanced_sample_indices(labels, maximum=40, seed=17)
    second = _balanced_sample_indices(labels, maximum=40, seed=17)

    assert np.array_equal(first, second)
    assert len(first) == 40
    selected = labels[first]
    assert (selected == "fruta").sum() == 20
    assert (selected == "verdura").sum() == 20


class _FixedModel:
    def __init__(self, classes: list[str], probabilities: list[float]) -> None:
        import numpy as np

        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, values):
        import numpy as np

        return np.repeat(self._probabilities.reshape(1, -1), len(values), axis=0)


def test_binary_baseline_reports_class_probabilities() -> None:
    model = BinaryBaseline(
        feature_config=FeatureConfig(),
        classifier=_FixedModel(["fruta", "verdura"], [0.8, 0.2]),
    )
    prediction = model.predict_vector(__import__("numpy").zeros(10, dtype="float32"))
    assert prediction["prediction"] == "fruta"
    assert prediction["confidence"] == __import__("pytest").approx(0.8)
    assert prediction["probabilities"] == __import__("pytest").approx(
        {"fruta": 0.8, "verdura": 0.2}
    )
    assert prediction["uncertain"] is False


def test_binary_baseline_trains_and_writes_metrics(tmp_path: Path) -> None:
    rows = []
    colors = {
        "fruta": [(220, 30, 30), (240, 120, 20), (180, 20, 90)],
        "verdura": [(20, 150, 40), (40, 110, 30), (80, 160, 50)],
    }
    for class_id, palette in colors.items():
        for index in range(6):
            path = tmp_path / f"{class_id}_{index}.png"
            Image.new("RGB", (40, 40), palette[index % len(palette)]).save(path)
            rows.append(
                {
                    "path": str(path),
                    "split": "train" if index < 4 else "test",
                    "valid": "True",
                    "mapping_status": "approved",
                    "class_id": class_id,
                }
            )

    manifest = tmp_path / "dataset.csv"
    model_path = tmp_path / "baseline.joblib"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    model, metrics = train_binary_baseline(
        manifest,
        model_path,
        metrics_path,
        feature_config=FeatureConfig(image_size=32),
        max_per_class=None,
        seed=7,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["train_images"] == 8
    assert metrics["evaluation_images"] == 4
    assert set(model.predict_path(tmp_path / "fruta_4.png")["probabilities"]) == {
        "fruta",
        "verdura",
    }


def test_classical_suite_selects_without_using_test(tmp_path: Path) -> None:
    rows = []
    colors = {
        "fruta": [(230, 30, 25), (235, 110, 15), (190, 30, 100)],
        "verdura": [(20, 145, 35), (35, 105, 25), (70, 170, 55)],
    }
    for class_id, palette in colors.items():
        for index in range(9):
            path = tmp_path / f"suite_{class_id}_{index}.png"
            Image.new("RGB", (42, 42), palette[index % len(palette)]).save(path)
            split = "train" if index < 5 else "validation" if index < 7 else "test"
            rows.append(
                {
                    "path": str(path),
                    "split": split,
                    "valid": "True",
                    "mapping_status": "approved",
                    "class_id": class_id,
                }
            )

    manifest = tmp_path / "dataset.csv"
    model_path = tmp_path / "classic.joblib"
    metrics_path = tmp_path / "classic_metrics.json"
    selection_path = tmp_path / "selection.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    model, metrics = train_classical_suite(
        manifest,
        model_path,
        metrics_path,
        selection_path,
        feature_config=FeatureConfig(image_size=32),
        augmentations=0,
        seed=11,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert selection_path.exists()
    assert metrics["selected_model"] in set(pd.read_csv(selection_path)["model"])
    assert metrics["test"]["images"] == 4
    restored = load_classical_model(model_path)
    prediction = restored.predict_path(tmp_path / "suite_fruta_7.png")
    assert set(prediction["probabilities"]) == {"fruta", "verdura"}
    breakdown_path = tmp_path / "breakdown.csv"
    evaluation = evaluate_classical_model(
        manifest, model_path, "test", breakdown_path
    )
    assert evaluation["images"] == 4
    assert breakdown_path.exists()
    assert set(pd.read_csv(breakdown_path)["class_id"]) == {"fruta", "verdura"}

