"""Punto de entrada para un Hugging Face Space (Streamlit)."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("FRUTAS_PROJECT_ROOT", str(ROOT))
# El disco de un Space es efímero: no se ofrece guardar ejemplos de reentrenamiento.
os.environ.setdefault("FRUTAS_DISABLE_FEEDBACK", "1")


# Sin spinner: set_page_config debe ser la primera llamada visible de Streamlit.
@st.cache_resource(show_spinner=False)
def _download_model() -> str:
    return hf_hub_download(
        os.environ.get("FRUTAS_MODEL_REPO", "ml-escom/frutas-verduras-modelo"),
        "classical_model.joblib",
        revision=os.environ.get("FRUTAS_MODEL_REVISION") or None,
        local_dir=ROOT / "artifacts",
    )


_download_model()

from frutas.app import render_app  # noqa: E402

render_app()
