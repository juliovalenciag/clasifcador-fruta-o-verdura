from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from frutas.classical_training import load_classical_model
from frutas.feedback import (
    classify_uploaded_image,
    decode_uploaded_image,
    save_training_example,
)


def _project_root() -> Path:
    configured = os.environ.get("FRUTAS_PROJECT_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "configs" / "datasets.yaml").exists():
            return resolved
    return Path.cwd().resolve()


def _load_model_cached(path: str, modified_ns: int):
    del modified_ns
    return load_classical_model(Path(path))


def render_app() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - mensaje para instalación incompleta
        raise RuntimeError(
            'Falta Streamlit. Instale la interfaz con: pip install -e ".[app]"'
        ) from exc

    st.set_page_config(
        page_title="Fruta o verdura",
        page_icon="🍎 / 🥦",
        layout="centered",
    )
    root = _project_root()

    st.title("Clasificador de frutas y verduras")
    st.caption(
        "Machine Learning"
    )

    with st.sidebar:
        st.header("Configuración")
        model_text = st.text_input(
            "Modelo",
            value="artifacts/classical_model.joblib",
            help="Ruta absoluta o relativa a la raíz del proyecto.",
        )
        use_tta = st.checkbox(
            "Usar TTA de 4 vistas",
            value=True,
            help="Mejora la robustez ante encuadre e iluminación; tarda un poco más.",
        )
        st.caption(f"Proyecto: {root}")

    model_path = Path(model_text).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    if not model_path.exists():
        st.error(
            f"No se encontró el modelo: {model_path}. Descárguelo del Hub del equipo "
            "(`frutas-data hub pull-model`) o entrénelo con `frutas-data train-classic`."
        )
        if st.button("Descargar el modelo del equipo desde Hugging Face", type="primary"):
            from frutas.config import ProjectPaths
            from frutas.hub import pull_model

            with st.spinner("Descargando modelo…"):
                try:
                    pull_model(ProjectPaths.from_root(root), model_path.parent)
                except Exception as exc:
                    st.error(f"No fue posible descargar el modelo: {exc}")
                    return
            st.rerun()
        return

    cached_loader = st.cache_resource(show_spinner=False)(_load_model_cached)
    try:
        model = cached_loader(str(model_path.resolve()), model_path.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"No fue posible cargar el modelo: {exc}")
        return

    uploaded = st.file_uploader(
        "Sube una fotografía",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        help="Máximo 20 MB. Procura que aparezca un solo objeto principal.",
    )
    if uploaded is None:
        st.info("Selecciona una imagen para comenzar.")
        return

    contents = uploaded.getvalue()
    upload_key = hashlib.sha256(contents).hexdigest()
    try:
        preview = decode_uploaded_image(contents)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(preview, caption=uploaded.name, use_container_width=True)
    if st.button("Clasificar imagen", type="primary", use_container_width=True):
        with st.spinner("Extrayendo características y clasificando…"):
            try:
                _, prediction = classify_uploaded_image(contents, model, tta=use_tta)
            except ValueError as exc:
                st.error(str(exc))
                return
        st.session_state["frutas_prediction"] = (upload_key, bool(use_tta), prediction)

    stored = st.session_state.get("frutas_prediction")
    if not stored or stored[0] != upload_key or stored[1] != bool(use_tta):
        return
    prediction = stored[2]
    predicted_class = str(prediction["prediction"])
    confidence = float(prediction["confidence"])
    probabilities = prediction["probabilities"]

    st.subheader("Resultado")
    left, right = st.columns(2)
    left.metric("Clasificación", predicted_class.upper())
    right.metric("Confianza", f"{confidence:.1%}")
    st.progress(
        float(probabilities["fruta"]),
        text=f"Fruta: {float(probabilities['fruta']):.1%}",
    )
    st.progress(
        float(probabilities["verdura"]),
        text=f"Verdura: {float(probabilities['verdura']):.1%}",
    )
    if prediction["uncertain"]:
        st.warning(
            "Predicción incierta: conviene revisar el encuadre o confirmar manualmente."
        )
    else:
        st.success(f"El modelo clasifica la imagen como **{predicted_class}**.")
    st.caption(
        f"Modelo: {prediction['model']} · vistas: {prediction['tta_views']} · "
        f"umbral de decisión: {float(prediction['decision_threshold']):.3f}"
    )

    if os.environ.get("FRUTAS_DISABLE_FEEDBACK") == "1":
        return

    with st.expander("Usar esta imagen para un futuro reentrenamiento"):
        st.write(
            "La imagen solo se guardará si una persona confirma la etiqueta correcta. "
            "La predicción del modelo no se usa como etiqueta automática."
        )
        correct_class = st.selectbox(
            "Etiqueta correcta",
            options=["fruta", "verdura"],
            index=None,
            placeholder="Selecciona la etiqueta real",
        )
        confirmed = st.checkbox("Confirmo que revisé la etiqueta manualmente")
        save_disabled = correct_class is None or not confirmed
        if st.button(
            "Guardar ejemplo etiquetado",
            disabled=save_disabled,
            use_container_width=True,
        ):
            try:
                saved = save_training_example(
                    contents,
                    str(correct_class),
                    root / "data" / "raw" / "team_photos_train",
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if saved.created:
                    st.success(f"Ejemplo guardado en `{saved.path}`.")
                else:
                    st.info(f"La misma imagen ya estaba guardada en `{saved.path}`.")
                st.code("frutas-data all\nfrutas-data train-classic", language="bash")
                st.caption(
                    "Guardar no modifica el modelo abierto. El reentrenamiento es explícito "
                    "y conserva validation/test separados."
                )


def launch() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            'Falta Streamlit. Instale la interfaz con: pip install -e ".[app]"'
        ) from exc

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless=false",
        *sys.argv[1:],
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    render_app()
