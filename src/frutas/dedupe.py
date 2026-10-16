from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _hash_segments(value: int, chunk_count: int) -> list[tuple[int, int]]:
    """Divide 64 bits sin solapamiento para indexar vecinos de Hamming."""

    base_width, wider_chunks = divmod(64, chunk_count)
    offset = 0
    segments: list[tuple[int, int]] = []
    for index in range(chunk_count):
        width = base_width + (1 if index < wider_chunks else 0)
        mask = (1 << width) - 1
        segments.append((index, (value >> offset) & mask))
        offset += width
    return segments


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def mark_duplicates(
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    phash_distance: int = 4,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    valid = frame[(frame["valid"].str.lower() == "true") & (frame["phash"] != "")].copy()
    uf = UnionFind(valid["image_id"].tolist())

    for _, group in valid.groupby("sha256"):
        ids = group["image_id"].tolist()
        for image_id in ids[1:]:
            uf.union(ids[0], image_id)

    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in valid.sort_values("image_id").itertuples():
        hash_to_ids[row.phash].append(row.image_id)

    # Consultar una vez por hash único evita trabajo cuadrático cuando miles de
    # recortes visualmente iguales comparten el mismo pHash.
    int_to_phash = {int(phash, 16): phash for phash in hash_to_ids}
    # Si dos hashes difieren en como máximo d bits, al dividirlos en d + 1
    # segmentos al menos un segmento debe coincidir exactamente. El índice usa
    # esa propiedad para obtener candidatos sin recorrer todos los hashes.
    chunk_count = min(max(phash_distance, 0) + 1, 64)
    segment_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for phash_value in sorted(int_to_phash):
        phash = int_to_phash[phash_value]
        ids = hash_to_ids[phash]
        representative = ids[0]
        for image_id in ids[1:]:
            uf.union(representative, image_id)
        segments = _hash_segments(phash_value, chunk_count)
        candidates: set[int] = set()
        for segment in segments:
            candidates.update(segment_buckets[segment])
        for candidate_value in candidates:
            if (phash_value ^ candidate_value).bit_count() <= phash_distance:
                candidate_hash = int_to_phash[candidate_value]
                uf.union(representative, hash_to_ids[candidate_hash][0])
        for segment in segments:
            segment_buckets[segment].append(phash_value)

    group_members: dict[str, list[str]] = defaultdict(list)
    for image_id in valid["image_id"]:
        group_members[uf.find(image_id)].append(image_id)
    duplicate_groups = {
        image_id: f"dup_{min(members)[:16]}"
        for members in group_members.values()
        for image_id in members
    }
    frame["duplicate_group"] = frame["image_id"].map(duplicate_groups).fillna("")
    counts = frame["duplicate_group"].value_counts()
    frame["is_duplicate_candidate"] = frame["duplicate_group"].map(counts).fillna(0).astype(int) > 1

    conflict_groups: set[str] = set()
    approved = frame[frame["mapping_status"] == "approved"]
    for group_name, group in approved[approved["is_duplicate_candidate"]].groupby("duplicate_group"):
        class_conflict = group["class_id"].replace("", pd.NA).dropna().nunique() > 1
        if class_conflict:
            conflict_groups.add(group_name)
    frame["duplicate_label_conflict"] = frame["duplicate_group"].isin(conflict_groups)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report = frame[frame["is_duplicate_candidate"]].sort_values(["duplicate_group", "source", "path"])
    report.to_csv(report_path, index=False)
    print(
        f"Duplicados: {report['duplicate_group'].nunique()} grupos candidatos, "
        f"{len(conflict_groups)} con conflicto de etiqueta."
    )
    return frame

