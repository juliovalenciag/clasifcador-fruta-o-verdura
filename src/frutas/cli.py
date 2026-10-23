from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import build_audit_report
from .config import ProjectPaths
from .dedupe import mark_duplicates
from .manifest import build_manifest
from .splits import assign_splits


def _project_root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path.cwd().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="frutas-data",
        description="Pipeline auditable para clasificar imágenes como fruta o verdura.",
    )
    parser.add_argument("--root", help="Raíz del proyecto; por defecto el directorio actual.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="Descarga un dataset registrado directamente en data/raw."
    )
    download.add_argument("dataset", help="ID definido en configs/datasets.yaml.")
    download.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza una descarga previa o incompleta.",
    )

    prepare_lvis = subparsers.add_parser(
        "prepare-lvis",
        help="Convierte las cajas YOLO de LVIS en recortes para clasificación.",
    )
    prepare_lvis.add_argument("--margin", type=float, default=0.15)
    prepare_lvis.add_argument("--min-object-size", type=int, default=24)

    subparsers.add_parser("inventory", help="Valida imágenes y genera el manifiesto inicial.")

    dedupe = subparsers.add_parser("dedupe", help="Marca duplicados exactos y perceptuales.")
    dedupe.add_argument("--phash-distance", type=int, default=4)

    split = subparsers.add_parser("split", help="Crea train/validation/test sin fuga de duplicados.")
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--validation", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--reset",
        action="store_true",
        help="Descarta asignaciones previas y vuelve a crear todos los splits.",
    )

    subparsers.add_parser("audit", help="Genera un reporte Markdown del estado de los datos.")
    subparsers.add_parser("all", help="Ejecuta inventory, dedupe, split y audit.")

    extract = subparsers.add_parser(
        "extract-features",
        help=(
            "Extrae una sola vez las características de todas las imágenes elegibles "
            "y las guarda en data/features/<extractor_id>/ (incremental por SHA-256)."
        ),
    )
    extract.add_argument("--image-size", type=int, default=64)
    extract.add_argument("--workers", type=int, default=0, help="Procesos; 0 = automático.")
    extract.add_argument("--store", help="Directorio de la caché; por defecto data/features/<id>.")

    hub = subparsers.add_parser(
        "hub",
        help="Comparte modelos y cachés de características con Hugging Face Hub.",
    )
    hub_commands = hub.add_subparsers(dest="hub_command", required=True)
    hub_push_model = hub_commands.add_parser("push-model", help="Sube artifacts/classical_model.joblib.")
    hub_push_model.add_argument("--artifacts", default="artifacts")
    hub_push_model.add_argument("--message")
    hub_push_model.add_argument("--tag", help="Etiqueta opcional, p. ej. v2.2.0.")
    hub_pull_model = hub_commands.add_parser("pull-model", help="Descarga el modelo del equipo.")
    hub_pull_model.add_argument("--artifacts", default="artifacts")
    hub_pull_model.add_argument("--revision", help="Commit o etiqueta; por defecto main.")
    hub_push_features = hub_commands.add_parser("push-features", help="Sube una caché de características.")
    hub_push_features.add_argument("--store", help="Por defecto la caché del extractor de 64 px.")
    hub_push_features.add_argument("--message")
    hub_pull_features = hub_commands.add_parser("pull-features", help="Descarga una caché de características.")
    hub_pull_features.add_argument("--id", dest="store_id", help="extractor_id; por defecto el de 64 px.")
    hub_pull_features.add_argument("--revision")
    hub_commands.add_parser("list-features", help="Lista las cachés disponibles en el Hub.")

    audit_labels = subparsers.add_parser(
        "audit-labels",
        help=(
            "Lista imágenes de train cuya etiqueta contradice un modelo out-of-fold "
            "(platos, jugos, escenas mixtas) para revisión humana."
        ),
    )
    audit_labels.add_argument("--features", help="Caché; por defecto la de 64 px.")
    audit_labels.add_argument("--threshold", type=float, default=0.15)
    audit_labels.add_argument("--folds", type=int, default=3)
    audit_labels.add_argument("--output", default="artifacts/label_suspects.csv")
    audit_labels.add_argument(
        "--contact-sheets",
        default="artifacts/label_review",
        help="Carpeta para mosaicos de revisión (requiere imágenes); vacío los omite.",
    )
    apply_review = subparsers.add_parser(
        "apply-label-review",
        help="Agrega a configs/excluded_images.csv las filas revisadas con excluir=si.",
    )
    apply_review.add_argument("reviewed_csv")

    baseline_train = subparsers.add_parser(
        "train-baseline",
        help="Entrena el baseline binario clásico con HSV, HOG y LBP.",
    )
    baseline_train.add_argument("--image-size", type=int, default=64)
    baseline_train.add_argument("--max-per-class", type=int, default=1000)
    baseline_train.add_argument("--seed", type=int, default=42)
    baseline_train.add_argument("--model", default="artifacts/binary_baseline.joblib")
    baseline_train.add_argument("--metrics", default="artifacts/binary_baseline_metrics.json")

    baseline_predict = subparsers.add_parser(
        "predict-baseline",
        help="Predice si una imagen es fruta o verdura con el baseline.",
    )
    baseline_predict.add_argument("image")
    baseline_predict.add_argument("--model", default="artifacts/binary_baseline.joblib")

    classic_train = subparsers.add_parser(
        "train-classic",
        help=(
            "Compara modelos clásicos, selecciona con validation, reentrena y evalúa test."
        ),
    )
    classic_train.add_argument("--image-size", type=int, default=64)
    classic_train.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Límite por clase y split; 0 utiliza todas las imágenes.",
    )
    classic_train.add_argument(
        "--augmentations",
        type=int,
        default=0,
        help="Variaciones suaves creadas en memoria por imagen de train.",
    )
    classic_train.add_argument(
        "--max-per-source-label",
        type=int,
        default=0,
        help=(
            "Máximo por fuente y etiqueta original en train. 0 (por defecto) usa "
            "todas las imágenes de train."
        ),
    )
    classic_train.add_argument(
        "--eval-max-per-source-label",
        type=int,
        default=180,
        help=(
            "Máximo por fuente y etiqueta en validation y test; evita que una fuente "
            "masiva domine la selección y las métricas. 0 desactiva el límite."
        ),
    )
    classic_train.add_argument(
        "--svm-max-rows",
        type=int,
        default=14000,
        help=(
            "Máximo de filas balanceadas para cada SVM RBF con PCA; los demás modelos usan "
            "todas las filas. 0 desactiva el límite."
        ),
    )
    classic_train.add_argument(
        "--features",
        nargs="?",
        const="auto",
        help=(
            "Usa una caché de características en lugar de leer imágenes. Sin valor usa "
            "data/features/<id> del --image-size. Si no existe data/processed/dataset.csv "
            "se entrena con el manifiesto incluido en la caché (no requiere imágenes)."
        ),
    )
    classic_train.add_argument(
        "--no-exclusions",
        action="store_true",
        help="Ignora configs/excluded_images.csv (exclusiones de train por revisión manual).",
    )
    classic_train.add_argument("--uncertainty-threshold", type=float, default=0.60)
    classic_train.add_argument("--seed", type=int, default=42)
    classic_train.add_argument("--model", default="artifacts/classical_model.joblib")
    classic_train.add_argument("--metrics", default="artifacts/classical_metrics.json")
    classic_train.add_argument(
        "--selection", default="artifacts/model_selection.csv"
    )

    classic_predict = subparsers.add_parser(
        "predict-classic",
        help="Predice fruta o verdura sobre una imagen o directorio completo de fotos.",
    )
    classic_predict.add_argument(
        "image",
        help="Ruta a una imagen individual o a una carpeta con fotografías.",
    )
    classic_predict.add_argument("--model", default="artifacts/classical_model.joblib")
    classic_predict.add_argument(
        "--output",
        help="Ruta CSV opcional para exportar las predicciones por imagen.",
    )
    classic_predict.add_argument(
        "--no-tta",
        action="store_true",
        help="Desactiva el ensamble de 4 vistas en tiempo de prueba (TTA).",
    )

    classic_evaluate = subparsers.add_parser(
        "evaluate-classic",
        help="Evalúa el modelo y genera un desglose por etiqueta original.",
    )
    classic_evaluate.add_argument(
        "--split", choices=["validation", "test", "external"], default="test"
    )
    classic_evaluate.add_argument("--model", default="artifacts/classical_model.joblib")
    classic_evaluate.add_argument(
        "--output", default="artifacts/error_analysis_by_label.csv"
    )
    classic_evaluate.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help=(
            "Excluye una fuente de la evaluación; se puede repetir. Sirve para "
            "comparar contra un test de referencia sin la fuente nueva."
        ),
    )
    classic_evaluate.add_argument(
        "--no-tta",
        action="store_true",
        help="Evalúa una sola vista en lugar de las 4 vistas usadas en inferencia.",
    )

    noise_evaluate = subparsers.add_parser(
        "evaluate-noise",
        help="Evalúa la robustez del modelo ante 8 condiciones explícitas de ruido.",
    )
    noise_evaluate.add_argument(
        "--split", choices=["validation", "test", "external"], default="test"
    )
    noise_evaluate.add_argument("--model", default="artifacts/classical_model.joblib")
    noise_evaluate.add_argument(
        "--output", default="artifacts/noise_robustness.csv"
    )
    noise_evaluate.add_argument(
        "--max-per-source-label",
        type=int,
        default=50,
        help="Máximo de imágenes por fuente y etiqueta para el estrés de ruido (0 usa todas).",
    )
    noise_evaluate.add_argument(
        "--no-tta",
        action="store_true",
        help="Desactiva TTA durante la prueba de estrés.",
    )
    noise_evaluate.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    paths = ProjectPaths.from_root(_project_root(args.root))

    if args.command == "download":
        from .download import download_dataset

        download_dataset(paths, args.dataset, args.force)
    if args.command == "prepare-lvis":
        from .lvis import prepare_registered_lvis

        report = prepare_registered_lvis(
            paths,
            margin_ratio=args.margin,
            min_object_size=args.min_object_size,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.command in {"inventory", "all"}:
        build_manifest(paths)
    if args.command in {"dedupe", "all"}:
        if not paths.manifest.exists():
            parser.error("No existe manifest.csv; ejecute primero inventory.")
        distance = getattr(args, "phash_distance", 4)
        mark_duplicates(paths.manifest, paths.deduplicated_manifest, paths.duplicate_report, distance)
    if args.command in {"split", "all"}:
        if not paths.deduplicated_manifest.exists():
            parser.error("No existe manifest_deduplicated.csv; ejecute primero dedupe.")
        assign_splits(
            paths.deduplicated_manifest,
            paths.split_manifest,
            getattr(args, "train", 0.70),
            getattr(args, "validation", 0.15),
            getattr(args, "test", 0.15),
            getattr(args, "seed", 42),
            not getattr(args, "reset", False),
        )
    if args.command in {"audit", "all"}:
        if paths.split_manifest.exists():
            source = paths.split_manifest
        elif paths.deduplicated_manifest.exists():
            source = paths.deduplicated_manifest
        else:
            source = paths.manifest
        if not source.exists():
            parser.error("No existe un manifiesto para auditar.")
        build_audit_report(source, paths.audit_report)
    if args.command == "extract-features":
        from .classic_features import FeatureConfig
        from .feature_store import build_feature_store, default_store_dir

        if not paths.split_manifest.exists():
            parser.error("No existe data/processed/dataset.csv; ejecute primero all.")
        config = FeatureConfig(image_size=args.image_size)
        store_dir = (
            paths.root / args.store if args.store else default_store_dir(paths.root, config)
        )
        report = build_feature_store(
            paths.split_manifest, store_dir, config, workers=args.workers or None
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.command == "hub":
        from . import hub as hub_module

        if args.hub_command == "push-model":
            report = hub_module.push_model(
                paths, paths.root / args.artifacts, args.message, args.tag
            )
        elif args.hub_command == "pull-model":
            report = hub_module.pull_model(paths, paths.root / args.artifacts, args.revision)
        elif args.hub_command == "push-features":
            from .classic_features import FeatureConfig
            from .feature_store import default_store_dir

            store_dir = (
                paths.root / args.store
                if args.store
                else default_store_dir(paths.root, FeatureConfig())
            )
            report = hub_module.push_features(paths, store_dir, args.message)
        elif args.hub_command == "pull-features":
            report = hub_module.pull_features(paths, args.store_id, args.revision)
        else:
            report = {"feature_stores": hub_module.list_remote_feature_stores(paths)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.command == "audit-labels":
        from .classic_features import FeatureConfig
        from .feature_store import default_store_dir, load_feature_store
        from .label_audit import audit_train_labels, load_exclusions, write_contact_sheets

        store_dir = (
            paths.root / args.features
            if args.features
            else default_store_dir(paths.root, FeatureConfig())
        )
        store = load_feature_store(store_dir)
        manifest = (
            pd.read_csv(paths.split_manifest, dtype=str, keep_default_na=False)
            if paths.split_manifest.exists()
            else store.manifest
        )
        report = audit_train_labels(
            store,
            manifest,
            paths.root / args.output,
            folds=args.folds,
            probability_threshold=args.threshold,
            already_excluded=load_exclusions(paths.excluded_images),
        )
        if args.contact_sheets and "path" in manifest:
            sheets = write_contact_sheets(
                paths.root / args.output, paths.root / args.contact_sheets
            )
            report["contact_sheets"] = len(sheets)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.command == "apply-label-review":
        from .label_audit import apply_label_review

        report = apply_label_review(Path(args.reviewed_csv).resolve(), paths.excluded_images)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.command == "train-baseline":
        from .baseline import train_binary_baseline
        from .classic_features import FeatureConfig

        if not paths.split_manifest.exists():
            parser.error("No existe data/processed/dataset.csv; ejecute primero split.")
        _, metrics = train_binary_baseline(
            paths.split_manifest,
            paths.root / args.model,
            paths.root / args.metrics,
            FeatureConfig(image_size=args.image_size),
            args.max_per_class,
            args.seed,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.command == "predict-baseline":
        from .baseline import load_baseline

        model_path = paths.root / args.model
        if not model_path.exists():
            parser.error(f"No existe el modelo {model_path}; ejecute primero train-baseline.")
        prediction = load_baseline(model_path).predict_path(Path(args.image).resolve())
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
    if args.command == "train-classic":
        from .classic_features import FeatureConfig
        from .classical_training import train_classical_suite
        from .label_audit import load_exclusions

        feature_config = FeatureConfig(image_size=args.image_size)
        store = None
        manifest_path = paths.split_manifest
        if args.features:
            from .feature_store import default_store_dir, load_feature_store

            store_dir = (
                default_store_dir(paths.root, feature_config)
                if args.features == "auto"
                else paths.root / args.features
            )
            store = load_feature_store(store_dir)
            feature_config = store.feature_config
            if not manifest_path.exists():
                manifest_path = store_dir / "manifest.csv"
                print(f"Entrenando solo con la caché: {store_dir}")
        if not manifest_path.exists():
            parser.error("No existe data/processed/dataset.csv; ejecute primero all.")
        _, metrics = train_classical_suite(
            manifest_path=manifest_path,
            model_path=paths.root / args.model,
            metrics_path=paths.root / args.metrics,
            selection_path=paths.root / args.selection,
            feature_config=feature_config,
            max_per_class=args.max_per_class,
            max_per_source_label=args.max_per_source_label,
            eval_max_per_source_label=args.eval_max_per_source_label,
            svm_max_rows=args.svm_max_rows,
            augmentations=args.augmentations,
            uncertainty_threshold=args.uncertainty_threshold,
            seed=args.seed,
            feature_store=store,
            excluded_train_sha256=None if args.no_exclusions else load_exclusions(
                paths.excluded_images
            ),
        )
        summary = {
            "selected_model": metrics["selected_model"],
            "test": metrics["test"],
            "external": metrics["external"],
            "model_path": str((paths.root / args.model).resolve()),
            "metrics_path": str((paths.root / args.metrics).resolve()),
            "selection_path": str((paths.root / args.selection).resolve()),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.command == "predict-classic":
        from .classical_training import predict_target

        model_path = paths.root / args.model
        if not model_path.exists():
            parser.error(f"No existe el modelo {model_path}; ejecute primero train-classic.")
        output_path = (paths.root / args.output) if args.output else None
        prediction = predict_target(
            target_path=Path(args.image),
            model_path=model_path,
            output_path=output_path,
            tta=not args.no_tta,
        )
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
    if args.command == "evaluate-classic":
        from .classical_training import evaluate_classical_model

        model_path = paths.root / args.model
        if not model_path.exists():
            parser.error(f"No existe el modelo {model_path}; ejecute primero train-classic.")
        if not paths.split_manifest.exists():
            parser.error("No existe data/processed/dataset.csv; ejecute primero all.")
        result = evaluate_classical_model(
            paths.split_manifest,
            model_path,
            args.split,
            paths.root / args.output,
            excluded_sources=tuple(args.exclude_source),
            tta=not args.no_tta,
        )
        summary = {key: value for key, value in result.items() if key != "breakdown"}
        summary["breakdown_path"] = str((paths.root / args.output).resolve())
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.command == "evaluate-noise":
        from .classical_training import evaluate_noise_robustness

        model_path = paths.root / args.model
        if not model_path.exists():
            parser.error(f"No existe el modelo {model_path}; ejecute primero train-classic.")
        if not paths.split_manifest.exists():
            parser.error("No existe data/processed/dataset.csv; ejecute primero all.")
        result = evaluate_noise_robustness(
            manifest_path=paths.split_manifest,
            model_path=model_path,
            split=args.split,
            output_path=paths.root / args.output,
            max_per_source_label=args.max_per_source_label,
            tta=not args.no_tta,
            seed=args.seed,
        )
        result["output_path"] = str((paths.root / args.output).resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()

