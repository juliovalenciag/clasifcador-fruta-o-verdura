---
title: Fruta o verdura
emoji: 🥦
colorFrom: green
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

Demo web del clasificador clásico fruta/verdura. El modelo se descarga del repositorio
de modelo del equipo en Hugging Face Hub al iniciar el Space.

Variables del Space (Settings → Variables and secrets):

- `FRUTAS_MODEL_REPO`: repositorio del modelo, p. ej. `mi-org/frutas-verduras-modelo`.
- `FRUTAS_MODEL_REVISION` (opcional): etiqueta o commit fijo, p. ej. `v2.2.0`.
- `HF_TOKEN` (secreto): solo si el repositorio del modelo es privado; basta permiso *read*.
