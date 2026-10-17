# Clasificador binario de frutas y verduras (v2.1.0)

Proyecto de Machine Learning clásico que recibe una imagen (o una carpeta completa de fotografías) y responde únicamente una de dos clases bajo **criterio botánico estricto**: **fruta** o **verdura**. Las especies y variedades se conservan exclusivamente como metadatos de auditoría y análisis de errores.

- **Fruta (criterio botánico):** estructura desarrollada a partir del ovario de la flor (incluye manzana, plátano, naranja, tomate, pepino, berenjena, pimiento, chile, calabaza, calabacita, maíz y vainas de leguminosas).
- **Verdura (criterio botánico):** raíz, tubérculo, bulbo, rizoma, tallo, hoja o flor comestible (incluye zanahoria, papa, camote, betabel, rábano, cebolla, ajo, puerro, jengibre, espárrago, apio, lechuga, espinaca, col, brócoli, coliflor y alcachofa).
- **Excluidos del alcance binario (`rejected`):** hongos (`mushroom`), semillas secas (`almond`, `chickpea`, `peanuts`) y alimentos procesados o empaquetados (`pickle`, `tofu`, jugos, lácteos).

La solución es **100 % Machine Learning clásico** (sin redes neuronales). En `v2.0.0`, cada imagen se transforma en un vector determinista de **1,405 características** (`float32`) balanceado por familias e invarianzas físicas:
- **Color, cromaticidad, índices botánicos y separación figura-fondo (839 dims — 59.7 %):** histogramas `1D` (`16` bins) de **HSV** y **CIE Lab** (global + centro), histogramas (`12` bins) de **4 índices botánicos/radiométricos** ($ExG = 2G - R - B$, $R-G$, $Y-B$, croma CIE $C^*$), histogramas de matiz (`Hue`) ponderados por saturación (suprimen fondos blancos o grises), histogramas conjuntos `2D` de cromaticidad `(H, S)` (`8x4`) y `(a*, b*)` (`6x6`), perfiles en **3 anillos radiales concéntricos** con contraste interior-borde, **segmentación automática figura-fondo** en espacio `Lab + Croma` (con momentos geométricos de excentricidad, compacidad y aspecto) y cuadrícula espacial `3x3`.
- **Forma, energía de bordes y HOG multiescala agrupado (458 dims — 32.6 %):** energía de gradiente **Sobel** y alta frecuencia **Laplaciana** en anillos concéntricos y cuadrícula `4x4` (`44` dims) + **HOG multiescala** agrupado en `3x3` macro-regiones (media y máximo = `162` dims), celdas centrales `5x5` del objeto (`225` dims) y resumen global invariante a traslación (`27` dims).
- **Textura LBP multiescala (108 dims — 7.7 %):** histogramas **LBP uniforme** en **3 escalas radiales** (`P=8, R=1` micro-textura; `P=16, R=2` meso-textura; `P=24, R=3` macro-textura) tanto globales como centrales.
- **Suite de 8 modelos clásicos + Ensamble ponderado (`soft_voting_top3`):** combina **Histogram Gradient Boosting** (`HistGradientBoostingClassifier` de profundidad 63 y 95), **SVM RBF con proyección `AdaptivePCA(256)` y `BlockBalancedScaler`** (`C=10` y `C=25` sobre `14,000` filas balanceadas), **Extra Trees** (`sqrt` y `0.40`) y clasificadores lineales regularizados.
- **Filtrado automático de calidad en escenas complejas (`LVIS`):** descarta parches diminutos (`< 64x64` px) y recortes provenientes de escenas mixtas (`fruta + verdura` simultáneas) o saturadas (`> 8` objetos por imagen).
- **Inferencia robusta (TTA):** promedio de **4 vistas en tiempo de prueba** (original, espejo horizontal, recorte central `86 %` y normalización de contraste `autocontrast`).

---

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks,app]"

# Descarga reproducible de las fuentes activas (5 en disco + fruits360 opcional)
frutas-data download kritikseth36
frutas-data download fruits_youssef
frutas-data download lvis_fruits_vegetables
frutas-data prepare-lvis
frutas-data download grocery_store
frutas-data download vegetable_ahmed
# Opcional (6.ª fuente curada con sparse-checkout 44 clases balanceadas):
# frutas-data download fruits360

# Inventario, deduplicación SHA-256 + pHash, partición sin fugas y auditoría
frutas-data all

# Entrenamiento y selección del modelo clásico final (v2.0.0)
frutas-data train-classic
```

Si Kaggle solicita autenticación (`kritikseth36`, `fruits_youssef` o `vegetable_ahmed`), configure su token personal en `export KAGGLE_API_TOKEN="..."`; nunca lo guarde dentro del repositorio. `lvis_fruits_vegetables` (Hugging Face), `grocery_store` (GitHub) y `fruits360` (GitHub sparse-checkout) se descargan sin requerir token de Kaggle.

---

## Guía rápida de uso y pruebas reales

### Interfaz web local

La manera más sencilla de probar el modelo es:

```bash
frutas-app
```

Se abrirá una página local en el navegador para subir una imagen, verla, clasificarla
con TTA y consultar las probabilidades de `fruta` y `verdura`. La sección opcional
**Usar esta imagen para un futuro reentrenamiento** permite guardar ejemplos únicamente
después de que una persona confirme la etiqueta correcta.

Las imágenes confirmadas se guardan en `data/raw/team_photos_train/fruta/` o
`data/raw/team_photos_train/verdura/`, deduplicadas por contenido. No modifican el
modelo inmediatamente. Para incorporarlas de forma auditable:

```bash
frutas-data all
frutas-data train-classic
```

Esta fuente tiene `fixed_split: train`: el feedback de la interfaz nunca se utiliza
para seleccionar modelos ni para reportar métricas de validation/test.

### Línea de comandos

El clasificador también puede probarse de **4 maneras**:

### 1. Probar una sola foto (tomada con celular o descargada)
```bash
frutas-data predict-classic ruta/a/mi_foto.jpg
```
Devuelve en JSON la clase predicha (`fruta` o `verdura`), la confianza (`0.0` a `1.0`), las probabilidades por clase, el umbral de decisión calibrado (`0.485`), el indicador `"uncertain": true/false` (si la confianza es menor a `0.60`) y las `4` vistas TTA utilizadas.

### 2. Probar una carpeta completa de fotos por lotes
```bash
frutas-data predict-classic ruta/a/carpeta_de_fotos/ --output artifacts/mis_predicciones.csv
```
- Clasifica todas las imágenes dentro de la carpeta (y subcarpetas) aplicando TTA.
- Si organizas tus fotos en subcarpetas `fruta/` y `verdura/` (o con nombres de categorías conocidas como `Carrots`, `Garlic`, `Tomato`, `Apple`), calcula **automáticamente** `accuracy`, `balanced_accuracy`, `f1_macro`, `accuracy_when_confident` y la matriz de confusión del lote.

### 3. Evaluar el conjunto oficial de fotos externas del equipo
Coloca las fotos tomadas por el equipo en:
- `data/external/team_photos/fruta/`
- `data/external/team_photos/verdura/`

Y ejecuta:
```bash
frutas-data all
frutas-data evaluate-classic --split external --output artifacts/external_breakdown.csv
```

### 4. Ejecutar el benchmark de estrés ante 8 condiciones de ruido
```bash
frutas-data evaluate-noise --split test --max-per-source-label 15 --output artifacts/noise_robustness.csv
```
Evalúa el modelo sobre imágenes limpias (`clean`) y bajo 7 perturbaciones explícitas (`gaussian_noise`, `gaussian_blur`, `illumination_low`, `illumination_high`, `rotation_crop`, `occlusion_patch`, `combined_heavy`), reportando la retención de `F1 macro`.

---

## Estado actual verificado (`v2.1.0` — modelo `v2.0.0` + interfaz de pruebas)

- **Datos auditados en disco:** `193,708` registros, `193,706` imágenes válidas, `0` etiquetas sin mapear y **`0` grupos duplicados con fuga entre particiones**.
- **Modelo ganador:** `soft_voting_top3` (ensamble ponderado de `hgb_depth_95`, `hgb_depth_63`, `svm_rbf_c10` y `svm_rbf_c25`), con umbral de decisión `0.485` seleccionado exclusivamente en `validation` (`89.80 %` accuracy y `0.8766` F1 macro en `validation`).
- **Desempeño en `test` multifuente muestreado (`9,293` imágenes):**
  - **Accuracy:** **`89.65 %`** (`0.8965`, **+8.39 pts** vs `81.26 %` en `v1.1.0`)
  - **Balanced Accuracy:** **`87.06 %`** (`0.8706`, **+6.99 pts** vs `80.07 %` en `v1.1.0`)
  - **Precision / Recall `fruta`:** **`92.95 %`** / **`92.78 %`** (`F1 = 0.9287`)
  - **Precision / Recall `verdura`:** **`80.95 %`** / **`81.34 %`** (`F1 = 0.8114`)
  - **F1 Macro:** **`0.8701`** (**+9.33 pts** vs `0.7768` en `v1.1.0`)
  - **Accuracy en predicciones confiables (`confianza >= 0.60`, cobertura `92.60 %`):** **`91.97 %`**
- **Desempeño en `test` elegible completo con TTA (`13,141` imágenes):**
  - **Accuracy global:** **`89.13 %`** (`F1 Macro = 0.8523`, `Accuracy confiable = 91.98 %`).
  - **Exactitud por fuente en `test`:** `vegetable_ahmed` **`97.69 %`** (`2,165` imgs), `grocery_store` **`91.17 %`** (`1,665` fotos reales de supermercado), `fruits_youssef` **`86.97 %`** (`9,119` imgs), `kritikseth36` **`79.55 %`** (`44` imgs) y `lvis_fruits_vegetables` **`77.03 %`** (`148` recortes limpios).
- **Ablación justa sin `vegetable_ahmed` (`10,976` imágenes de las 4 fuentes previas):** accuracy **`86.63 %`**, balanced accuracy **`80.34 %`** y F1 macro **`0.8045`** (frente a `80.34 %`, `74.55 %` y `0.7258` en `v1.1.0`).
- **Robustez ante ruido (`evaluate-noise`, 8 condiciones sobre `1,677` imágenes de `test`):**
  - **Exactitud en limpio (`clean`):** **`87.00 %`** (`F1 limpio = 0.8303`, +10.82 pts sobre `v1.1.0`).
  - **Retención media de F1 Macro bajo ruido:** **`95.58 %`** (`F1 medio con ruido = 0.7936`, superando incluso el desempeño sin ruido de versiones previas).

Artefactos generados localmente en `artifacts/`:
- `artifacts/classical_model.joblib`: modelo `v2.0.0` (`soft_voting_top3`) verificado y usado por defecto.
- `artifacts/classical_metrics.json`: métricas completas de validación (`9,422` imgs) y test (`9,293` imgs).
- `artifacts/model_selection.csv`: tabla comparativa de los 8 candidatos clásicos y el ensamble.
- `artifacts/error_analysis_by_label.csv`: desglose de exactitud en las 145 combinaciones `(fuente, etiqueta)` de `test` (`13,141` imgs).
- `artifacts/noise_robustness.csv`: resultados de las 8 condiciones del benchmark de estrés ante ruido.

La guía operativa detallada está en [guia-equipo.md](guia-equipo.md) y el informe técnico completo en [reporte-estado-proyecto.md](reporte-estado-proyecto.md).
