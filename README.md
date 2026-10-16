# Clasificador binario de frutas y verduras

Proyecto de Machine Learning que recibe una imagen y responde únicamente una de
dos clases: **fruta** o **verdura**. Las especies y variedades no son objetivos del
modelo; se conservan solamente como información de auditoría cuando el dataset las
proporciona.

El proyecto adopta el **criterio botánico**:
tomate, pepino, berenjena, chile, pimiento, calabaza, maíz y vainas de leguminosas se
etiquetan como fruta; raíces, tubérculos, bulbos, tallos, hojas y flores se etiquetan
como verdura.

La solución usa Machine Learning clásico, sin redes neuronales. Cada imagen se
convierte en un vector numérico de **2,068 características** mediante descriptores de
color **HSV y CIE Lab** (globales, del recorte central y momentos espaciales en
cuadrícula), forma **HOG** y textura **LBP uniforme** (global y central), ponderados
con escalado balanceado por bloques (`BlockBalancedScaler`) para que color, forma y
textura aporten varianza equivalente. El pipeline no modifica ni elimina imágenes
originales: genera manifiestos, grupos de duplicados, particiones, reportes y
artefactos de entrenamiento auditables.

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

1. Validación e inventario de imágenes (`167,287` registros en las 3 fuentes activas).
2. Detección de duplicados exactos (`SHA-256`) y perceptuales (`pHash`).
3. Partición agrupada y estratificada en `train`/`validation`/`test` sin fuga de duplicados.
4. Reporte de calidad en `data/interim/data_report.md`.

## Entrenamiento clásico completo

Después de aprobar el reporte de datos:

```bash
frutas-data train-classic --image-size 64 --augmentations 1
frutas-data predict-classic ruta/a/imagen.jpg
frutas-data evaluate-classic --split test
```

El comando compara regresión logística (`C=1`, `C=10`), SGD con pérdida logística,
SVM con kernel RBF calibrado (`CalibratedClassifierCV` con `C=10`, `C=30`), Extra Trees
regularizados (`min_samples_leaf=5`) y un ensamble de votación suave (`soft_voting_top3`).
Selecciona exclusivamente con `validation` según **F1 macro**, reentrena con
`train + validation` y evalúa `test` una sola vez. Los aumentos con ruido se crean en
memoria solamente para `train`.

Por defecto se usan hasta 100 imágenes por combinación fuente/etiqueta en cada
split. SVM RBF recibe hasta 6,000 filas balanceadas porque su costo crece mucho más
rápido; regresión logística, SGD y Extra Trees aprovechan todas las filas
seleccionadas. Ambos límites se pueden cambiar con `--max-per-source-label` y
`--svm-max-rows`.

### Estado actual verificado (v0.6.0 — 3 datasets activos)

- **Datos auditados:** `167,287` registros (`kritikseth36`, `fruits_youssef` y `lvis_fruits_vegetables`), `0` fugas de duplicados entre particiones.
- **Modelo seleccionado en validación:** `soft_voting_top3` (`F1 macro = 0.7169`, `Balanced Accuracy = 0.7252`, `Accuracy = 0.7606`).
- **Desempeño en test interno (`4,718` imágenes muestreadas):**
  - **Accuracy:** `76.37%` (`0.7637`)
  - **Balanced Accuracy:** `72.81%` (`0.7281`)
  - **F1 Macro:** `71.27%` (`0.7127`)
  - **Accuracy en predicciones confiables (`confianza >= 0.60`):** `81.75%` (`0.8175`, cobertura `76.41%`)

Resultados locales:

- `artifacts/classical_model.joblib`: pipeline listo para predecir.
- `artifacts/classical_metrics.json`: métricas de validación, test y test externo.
- `artifacts/model_selection.csv`: comparación de candidatos.
- `artifacts/error_analysis_by_label.csv`: desglose de exactitud por fuente y etiqueta original.

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

La guía operativa completa está en [guia-equipo.md](guia-equipo.md) y el análisis de
estado detallado en [reporte-estado-proyecto.md](reporte-estado-proyecto.md).
