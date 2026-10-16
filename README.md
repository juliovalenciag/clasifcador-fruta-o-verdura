# Clasificador binario de frutas y verduras

Proyecto de Machine Learning que recibe una imagen y responde únicamente una de
dos clases: **fruta** o **verdura**. Las especies y variedades no son objetivos del
modelo; se conservan solamente como información de auditoría cuando el dataset las
proporciona.

El proyecto adopta el **criterio botánico**:
tomate, pepino, berenjena, chile, pimiento y maíz se etiquetan como fruta; raíces,
tubérculos, bulbos, tallos, hojas y flores se etiquetan como verdura.

La solución usa Machine Learning clásico, sin redes neuronales. Cada imagen se
convierte en un vector numérico mediante HSV, HOG y LBP. El pipeline no modifica ni
elimina imágenes originales: genera manifiestos, grupos de duplicados, particiones,
reportes y artefactos de entrenamiento auditables.

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"

frutas-data download kritikseth36
frutas-data download fruits_youssef
frutas-data download lvis_fruits_vegetables
frutas-data prepare-lvis
frutas-data all
frutas-data train-classic
```

Si Kaggle solicita autenticación, configure un token personal en
`KAGGLE_API_TOKEN`; nunca lo guarde en el repositorio.

El comando `all` ejecuta:

1. Validación e inventario de imágenes.
2. Detección de duplicados exactos y perceptuales.
3. Partición agrupada y estratificada en train/validation/test.
4. Reporte de calidad en `data/interim/data_report.md`.

## Entrenamiento clásico completo

Después de aprobar el reporte de datos:

```bash
frutas-data train-classic --image-size 64 --augmentations 1
frutas-data predict-classic ruta/a/imagen.jpg
frutas-data evaluate-classic --split test
```

El comando compara regresión logística, SGD, SVM con kernel RBF, Extra Trees y un
ensamble de votación suave. Selecciona exclusivamente con `validation`, reentrena
con `train + validation` y evalúa `test` una sola vez. Los aumentos con ruido se
crean en memoria solamente para `train`.

Por defecto se usan hasta 100 imágenes por combinación fuente/etiqueta en cada
split. SVM RBF recibe hasta 6,000 filas balanceadas porque su costo crece mucho más
rápido; regresión logística, SGD y Extra Trees aprovechan todas las filas
seleccionadas. Ambos límites se pueden cambiar con `--max-per-source-label` y
`--svm-max-rows`.

Resultados locales:

- `artifacts/classical_model.joblib`: pipeline listo para predecir.
- `artifacts/classical_metrics.json`: métricas de validación, test y test externo.
- `artifacts/model_selection.csv`: comparación de candidatos.

`train-baseline` continúa disponible como comprobación rápida y simple, pero el
modelo final se genera con `train-classic`.

## Datos y GitHub

Los datasets, matrices de características y modelos no se suben a GitHub. El
repositorio contiene código, configuraciones, notebooks y documentación. Los datos
pesados se descargan con scripts. Google Drive se reserva exclusivamente para las
imágenes y datasets compartidos del equipo.

`LVIS_Fruits_And_Vegetables` requiere `prepare-lvis` porque sus etiquetas originales
son cajas YOLO. El comando genera recortes con contexto en `data/derived/`; nunca
modifica las imágenes descargadas en `data/raw/`.

La guía operativa completa está en [guia-equipo.md](guia-equipo.md).
