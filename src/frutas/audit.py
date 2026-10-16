from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Sin registros._"
    return frame.to_markdown(index=False)


def build_audit_report(manifest_path: Path, output_path: Path) -> str:
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    valid = frame["valid"].str.lower() == "true"
    conflicts = frame.get("duplicate_label_conflict", pd.Series(False, index=frame.index)).astype(str).str.lower() == "true"
    duplicate_candidates = frame.get("is_duplicate_candidate", pd.Series(False, index=frame.index)).astype(str).str.lower() == "true"

    source_summary = (
        frame.groupby("source", as_index=False)
        .agg(
            images=("image_id", "size"),
            valid=("valid", lambda values: values.astype(str).str.lower().eq("true").sum()),
            labels=("original_label", "nunique"),
            approved=("mapping_status", lambda values: values.eq("approved").sum()),
        )
        .sort_values("images", ascending=False)
    )
    mapping_gaps = (
        frame[valid & (frame["mapping_status"] != "approved")]
        .groupby(["source", "original_label", "mapping_status"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    class_distribution = (
        frame[frame.get("split", "") .isin(["train", "validation", "test", "external"])]
        .groupby(["class_id", "split"], as_index=False)
        .size()
        .sort_values(["class_id", "split"])
    ) if "split" in frame.columns else pd.DataFrame()
    source_class_distribution = (
        frame[frame["split"].isin(["train", "validation", "test", "external"])]
        .groupby(["source", "class_id", "split"], as_index=False)
        .size()
        .sort_values(["source", "class_id", "split"])
    ) if "split" in frame.columns else pd.DataFrame()
    original_label_distribution = (
        frame[frame["split"].isin(["train", "validation", "test", "external"])]
        .groupby(["source", "original_label", "class_id", "split"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    ) if "split" in frame.columns else pd.DataFrame()

    leakage = pd.DataFrame()
    if "split" in frame.columns and "duplicate_group" in frame.columns:
        leakage = (
            frame[(frame["duplicate_group"] != "") & frame["split"].isin(["train", "validation", "test"])]
            .groupby("duplicate_group", as_index=False)
            .agg(number_of_splits=("split", "nunique"), splits=("split", lambda x: ",".join(sorted(set(x)))))
        )
        leakage = leakage[leakage["number_of_splits"] > 1]

    report = f"""# Reporte de auditoría de datos

Generado a partir de `{manifest_path.name}`. Este reporte no modifica imágenes.

## Resumen

- Registros: **{len(frame)}**
- Imágenes válidas: **{int(valid.sum())}**
- Imágenes inválidas: **{int((~valid).sum())}**
- Candidatos duplicados: **{int(duplicate_candidates.sum())}**
- Conflictos de etiqueta entre duplicados: **{int(conflicts.sum())}**
- Grupos con fuga entre particiones: **{len(leakage)}**

## Fuentes

{_table(source_summary)}

## Etiquetas pendientes de normalizar

{_table(mapping_gaps.head(300))}

## Distribución por categoría y partición

{_table(class_distribution.head(1000))}

## Distribución por fuente, clase y partición

{_table(source_class_distribution.head(1000))}

## Etiquetas originales representadas

Esta tabla sirve para detectar si una sola especie o una sola fuente domina una clase,
aunque la especie no sea el objetivo del modelo.

{_table(original_label_distribution.head(500))}

## Fugas detectadas

{_table(leakage.head(200))}

## Criterio de avance

No se debe entrenar el modelo final mientras existan conflictos de etiqueta entre
duplicados, fugas entre particiones, una clase ausente o una fuente que identifique
por sí sola a una de las dos clases.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Reporte guardado en {output_path}")
    return report

