from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _allocate_counts(number_of_groups: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Distribuye grupos; con tres o más conserva las tres particiones."""
    if number_of_groups == 1:
        return 1, 0, 0
    if number_of_groups == 2:
        return 1, 0, 1

    counts = np.ones(3, dtype=int)
    remaining = number_of_groups - 3
    raw = np.asarray(fractions, dtype=float) * remaining
    extra = np.floor(raw).astype(int)
    counts += extra
    missing = remaining - int(extra.sum())
    order = np.argsort(-(raw - extra))
    for index in order[:missing]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _stratified_group_assignment(
    groups: pd.DataFrame,
    fractions: tuple[float, float, float],
    random_state: int,
) -> dict[str, str]:
    rng = np.random.default_rng(random_state)
    assigned: dict[str, str] = {}
    for _, stratum in groups.groupby("stratum", sort=True):
        group_ids = stratum["group_id"].to_numpy(copy=True)
        rng.shuffle(group_ids)
        n_train, n_validation, _ = _allocate_counts(len(group_ids), fractions)
        train_end = n_train
        validation_end = n_train + n_validation
        assigned.update({group_id: "train" for group_id in group_ids[:train_end]})
        assigned.update(
            {group_id: "validation" for group_id in group_ids[train_end:validation_end]}
        )
        assigned.update({group_id: "test" for group_id in group_ids[validation_end:]})
    return assigned


def assign_splits(
    manifest_path: Path,
    output_path: Path,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
    preserve_existing: bool = True,
) -> pd.DataFrame:
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Las fracciones train/validation/test deben sumar 1.")
    previous_by_image: dict[str, str] = {}
    if preserve_existing and output_path.exists():
        previous = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        if {"image_id", "split"}.issubset(previous.columns):
            previous = previous[previous["split"].isin(["train", "validation", "test"])]
            previous_by_image = dict(zip(previous["image_id"], previous["split"]))

    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    frame["split"] = "excluded"
    frame["split_assignment"] = "excluded"
    frame["exclusion_reason"] = ""

    valid_mask = frame["valid"].str.lower() == "true"
    approved_mask = frame["mapping_status"] == "approved"
    conflict_mask = frame["duplicate_label_conflict"].str.lower() == "true"
    frame.loc[~valid_mask, "exclusion_reason"] = "invalid_image"
    frame.loc[valid_mask & ~approved_mask, "exclusion_reason"] = "mapping_not_approved"
    frame.loc[valid_mask & approved_mask & conflict_mask, "exclusion_reason"] = "duplicate_label_conflict"

    eligible = valid_mask & approved_mask & ~conflict_mask
    external = eligible & (frame["dataset_role"] == "external")
    frame.loc[external, "split"] = "external"
    frame.loc[external, "split_assignment"] = "external"

    candidates = frame[eligible & (frame["dataset_role"] == "trainable")].copy()
    if candidates.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print("No hay imágenes entrenables aprobadas; revise label_mapping.csv.")
        return frame

    candidates["group_id"] = candidates["duplicate_group"].where(
        candidates["duplicate_group"] != "", candidates["image_id"]
    )
    group_labels = (
        candidates.groupby("group_id", as_index=False)
        .agg(class_id=("class_id", lambda values: values.value_counts().index[0]))
    )
    group_labels["stratum"] = group_labels["class_id"]
    split_by_group: dict[str, str] = {}
    preserved_group_ids: set[str] = set()
    original_group_ids: set[str] = set()
    preserved_groups = 0
    merged_previous_groups = 0
    if previous_by_image:
        prior = candidates[["group_id", "image_id"]].copy()
        prior["previous_split"] = prior["image_id"].map(previous_by_image)
        prior = prior.dropna(subset=["previous_split"])
        if not prior.empty:
            counts = (
                prior.groupby(["group_id", "previous_split"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            priority = {"train": 0, "validation": 1, "test": 2}
            counts["priority"] = counts["previous_split"].map(priority)
            chosen = (
                counts.sort_values(
                    ["group_id", "count", "priority"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("group_id")
            )
            split_by_group = dict(zip(chosen["group_id"], chosen["previous_split"]))
            preserved_group_ids = set(split_by_group)
            preserved_groups = len(chosen)
            merged_previous_groups = int(
                counts.groupby("group_id")["previous_split"].nunique().gt(1).sum()
            )

    # Para fuentes con particiones publicadas, mantenerlas evita que distintos
    # recortes de una misma escena terminen en train y test. Si un grupo de
    # duplicados une particiones distintas, prevalece la mayoría y después la
    # prioridad determinista train/validation/test.
    remaining_ids = set(group_labels["group_id"]).difference(split_by_group)
    original = candidates[candidates["group_id"].isin(remaining_ids)].copy()
    original = original[
        original["original_split"].isin(["train", "validation", "test"])
    ]
    if not original.empty:
        original_counts = (
            original.groupby(["group_id", "original_split"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        priority = {"train": 0, "validation": 1, "test": 2}
        original_counts["priority"] = original_counts["original_split"].map(priority)
        original_chosen = (
            original_counts.sort_values(
                ["group_id", "count", "priority"],
                ascending=[True, False, True],
            )
            .drop_duplicates("group_id")
        )
        original_assignments = dict(
            zip(original_chosen["group_id"], original_chosen["original_split"])
        )
        split_by_group.update(original_assignments)
        original_group_ids = set(original_assignments)

    unassigned = group_labels[~group_labels["group_id"].isin(split_by_group)]
    new_assignments = _stratified_group_assignment(
        unassigned,
        (train_fraction, validation_fraction, test_fraction),
        random_state,
    )
    split_by_group.update(new_assignments)
    assigned = candidates["group_id"].map(split_by_group)
    frame.loc[candidates.index, "split"] = assigned
    preserved_ids = candidates["group_id"].isin(preserved_group_ids)
    original_ids = candidates["group_id"].isin(original_group_ids)
    frame.loc[candidates.index[preserved_ids], "split_assignment"] = "preserved"
    frame.loc[candidates.index[original_ids], "split_assignment"] = "original"
    frame.loc[
        candidates.index[~preserved_ids & ~original_ids], "split_assignment"
    ] = "new"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    counts = frame["split"].value_counts().to_dict()
    print(f"Particiones guardadas en {output_path}: {counts}")
    if preserved_groups:
        print(
            f"Grupos previos preservados: {preserved_groups}; "
            f"fusiones que exigieron unificar split: {merged_previous_groups}."
        )
    return frame

