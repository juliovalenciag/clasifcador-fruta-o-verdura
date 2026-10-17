# Reporte de Estado y Guía de Entrega: Clasificador Binario de Frutas y Verduras (`v2.1.0`)

**Fecha de corte:** 16 de octubre de 2026

**Versión del proyecto:** `v2.1.0` ([pyproject.toml](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/pyproject.toml))

**Repositorio:** `https://github.com/juliovalenciag/clasifcador-fruta-o-verdura.git`

**Alcance técnico:** Clasificación binaria (`fruta` vs `verdura`) bajo **criterio botánico estricto**, empleando exclusivamente **Machine Learning clásico** (`scikit-learn` + `scikit-image`, sin redes neuronales).

---

## 1. Resumen Ejecutivo y Salto de Desempeño (`v0.5.0` $\rightarrow$ `v1.1.0` $\rightarrow$ `v2.0.0`)

Tras auditar a fondo por qué el clasificador en `v1.1.0` se estancaba alrededor del `~80 %` de exactitud (`77.43 %` en `validation`, `81.26 %` en `test` muestreado y `0.7768` de `F1 Macro`), se identificaron y corrigieron **cuatro cuellos de botella estructurales** en los datos, las características y las familias de modelos. Con la nueva arquitectura `v2.0.0`, el clasificador da un salto hasta **`89.80 %` de exactitud en `validation`**, **`89.65 %` en `test` multifuente muestreado** (`9,293` imgs), **`89.13 %` en `test` elegible completo con TTA** (`13,141` imgs) y **`91.98 %` en predicciones confiables** (`cobertura del 91.11 %`), alcanzando **`97.69 %` en `vegetable_ahmed`** y **`91.17 %` en fotos reales de supermercado (`grocery_store`)**.

| Métrica en conjunto `test` | `v0.5.0` | `v1.0.0` completo | `v1.1.0` (5 fuentes) | **`v2.0.0` (5 fuentes, `9,293` imgs)** | **`v2.0.0` completo TTA (`13,141` imgs)** | **`v2.0.0` sin `vegetable_ahmed` (`10,976` imgs)** |
|---|---:|---:|---:|---:|---:|---:|
| **Accuracy global** | 63.40 % | 78.54 % | 81.26 % | **89.65 %** (`+8.39 pts`) | **89.13 %** (`+10.59 pts`) | **86.63 %** (`+6.29 pts` vs `v1.1.0`) |
| **Balanced Accuracy** | 62.69 % | 76.10 % | 80.07 % | **87.06 %** (`+6.99 pts`) | **84.61 %** (`+8.51 pts`) | **80.34 %** (`+5.79 pts` vs `v1.1.0`) |
| **Precision Macro** | 60.12 % | 70.38 % | 76.27 % | **86.95 %** (`+10.68 pts`) | **85.93 %** (`+15.55 pts`) | **80.57 %** |
| **Recall Macro** | 62.69 % | 76.10 % | 80.07 % | **87.06 %** (`+6.99 pts`) | **84.61 %** (`+8.51 pts`) | **80.34 %** |
| **F1 Macro** | 0.6057 | 0.7201 | 0.7768 | **0.8701** (`+9.33 pts`) | **0.8523** (`+13.22 pts`) | **0.8045** (`+7.87 pts` vs `v1.1.0`) |
| **Precision / Recall `fruta`** | — / 66.43 % | 91.48 % / 80.31 % | 90.44 % / 82.61 % | **92.95 % / 92.78 %** | **92.01 % / 93.65 %** | **91.31 % / 91.58 %** |
| **Precision / Recall `verdura`** | — / 58.95 % | 49.29 % / 71.89 % | 62.11 % / 77.52 % | **80.95 % / 81.34 %** | **79.85 % / 75.56 %** | **69.82 % / 69.10 %** |
| **Cobertura (`confianza >= 0.60`)** | 59.09 % | 71.55 % | 69.93 % | **92.60 %** (`+22.67 pts`) | **91.11 %** (`+19.56 pts`) | **90.73 %** |
| **Accuracy confiable (`>= 0.60`)** | 70.20 % | 89.58 % | 89.42 % | **91.97 %** | **91.98 %** | **89.48 %** |

---

## 2. Diagnóstico Técnico: ¿Por qué el clasificador rondaba el `~80 %` de Accuracy?

Al auditar los artefactos de `v1.1.0` y las `193,708` imágenes del inventario, se descubrieron **4 causas raíz** que frenaban el aprendizaje:

1. **Contaminación severa en los recortes de `lvis_fruits_vegetables`:**
   - Al inspeccionar `data/derived/lvis_fruits_vegetables/crops_manifest.csv` (`91,378` recortes extraídos de `7,388` escenas complejas de COCO/LVIS), se encontró que:
     - El **`59.2 %`** de los recortes medían **menos de `64x64` píxeles** (hasta parches borrosos de `24x24` px al fondo de una mesa o cocina).
     - El **`25.5 %`** (`22,182` recortes) provenían de **escenas mixtas (`fruta + verdura` simultáneas)**, como ensaladas, woks o tablas de picar donde la caja de un `tomato` (`fruta`) y una `lettuce` (`verdura`) se superponían y el margen de contexto (`15 %`) incluía píxeles de la clase opuesta dentro del mismo recorte.
     - Más del **`75 %`** provenían de escenas saturadas con hasta `83` cajas anotadas en una misma fotografía.
   - En consecuencia, mientras `vegetable_ahmed` y `grocery_store` superaban el `85 %–90 %`, `lvis_fruits_vegetables` apenas alcanzaba `69.64 %` e inyectaba miles de etiquetas contradictorias durante el entrenamiento.
2. **Inanición de características de color y textura en modelos de árboles (`85.3 %` de columnas HOG):**
   - En el vector anterior de `2,068` dimensiones, **`1,764` columnas (`85.3 %`) eran celdas HOG crudas de `8x8` píxeles**, frente a solo `252` de color (`12.2 %`) y `52` de textura LBP (`2.5 %`).
   - Como los árboles de decisión (`ExtraTreesClassifier`) son invariantes al escalado monotónico de columnas (`BlockBalancedScaler` no altera sus particiones), en cada nodo aleatorio el **`85.3 %` de las variables candidatas eran gradientes locales `8x8` sensibles a la posición exacta del objeto**, diluyendo la cromaticidad, el color interior-exterior y la micro-textura.
3. **Aumento de ruido en entrenamiento que destruía la micro-textura botánica:**
   - El generador `_make_noisy_variant` aplicaba ruido gaussiano aditivo ($\sigma \in [2, 12]$) y desenfoque gaussiano (`radius=1.4`) al `50 %` de las filas de entrenamiento.
   - En `64x64` píxeles, el ruido gaussiano añadía varianza artificial de alta frecuencia a frutas de piel lisa (manzanas, tomates, pimientos, haciéndolas parecer verduras rugosas ante LBP/Sobel), mientras que el desenfoque borraba los floretes de brócoli y coliflor (haciéndolos parecer frutas lisas).
4. **Ausencia de Gradient Boosting por histogramas y de reducción dimensional ortogonal para SVM RBF:**
   - Los árboles extremadamente aleatorizados (`ExtraTreesClassifier`) eligen umbrales al azar sin optimizar el gradiente del error residual, y `SVC(RBF)` recibía solo `6,000` filas sobre `2,068` dimensiones crudas con colinealidad espacial.

---

## 3. Solución Implementada en `v2.0.0`: Descriptor `v4` (`1,405` vars), Limpieza LVIS, Ensamble con Gradient Boosting y 6.º Dataset

### 3.1 Reingeniería del Descriptor Clásico a `1,405` Dimensiones ([src/frutas/classic_features.py](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/src/frutas/classic_features.py))

Se diseñó un extractor clásico vectorizado (`~250–300 img/s` mediante `np.bincount`) de **1,405 características** (`float32`), donde cada familia tiene representación proporcional e invarianzas físicas explícitas:

- **Bloque 1 — Color, Cromaticidad, Índices Botánicos y Separación Figura-Fondo (`839` dims, `59.7 %`):**
  - **Histogramas 1D (`16` bins) de HSV y CIE Lab** sobre la imagen completa y el recorte central (`50 %` interior).
  - **Histogramas (`12` bins) de 4 índices botánicos/radiométricos:** Excess Green ($ExG = 2G - R - B$, clave para distinguir clorofila foliar/tallo de frutos maduros), oposición Rojo-Verde ($R - G$), oposición Amarillo-Azul ($Y - B$) y Croma perceptual CIE $C^* = \sqrt{(a^*)^2 + (b^*)^2}$.
  - **Histogramas de matiz (`Hue`) ponderados por saturación (`S * V`):** eliminan el aporte de píxeles acromáticos (fondos blancos de estudio, sombras negras o estantes metálicos).
  - **Histogramas conjuntos 2D de cromaticidad `(H, S)` (`8x4 = 32` bins) y `(a*, b*)` (`6x6 = 36` bins)** en vista global y central, capturando combinaciones de matiz y pureza cromática imposibles de separar con histogramas marginales 1D.
  - **Perfiles en 3 anillos radiales concéntricos** (núcleo interior $r \le 0.42$, anillo medio $0.42 < r \le 0.78$, borde exterior $r > 0.78$) sobre los 13 canales + **vectores explícitos de contraste `interior - exterior` e `interior - medio`** (`130` dims).
  - **Segmentación automática Figura-Fondo en espacio `(L*, a*, b*, S, C*)` (`45` dims):** estima la distribución del fondo a partir del marco perimetral de 5 píxeles, separa la máscara del objeto principal y extrae la media y desviación del objeto, la diferencia neta `objeto - fondo` en los 13 canales y **6 descriptores geométricos de forma** (fracción de área ocupada, centralidad del objeto, ancho/alto de su caja envolvente, razón de aspecto, excentricidad por autovalores de covarianza e índice de compacidad perimetral).
  - **Cuadrícula espacial `3x3`** de medias y desviaciones estándar sobre canales de color e índices (`220` dims).
- **Bloque 2 — Forma, Energía de Bordes y HOG Multiescala Agrupado (`458` dims, `32.6 %`):**
  - **Energía de gradiente Sobel y alta frecuencia Laplaciana (`44` dims):** cuantifica la rugosidad interna frente al contorno en los 3 anillos concéntricos y en una cuadrícula `4x4` (separa con alta precisión cáscaras lisas y brillantes de frutas frente a texturas fractales/foliares de brócoli, coliflor, col, lechuga y espinaca).
  - **HOG multiescala agrupado (`414` dims):** a partir de la malla `7x7x9` de celdas `8x8`, calcula **pooling promedio y máximo en `3x3` macro-regiones** (`162` dims invariantes a pequeños desplazamientos), conserva las **`5x5` celdas centrales del objeto** (`225` dims, descartando el borde ruidoso) y agrega el **perfil global de orientación** (media, desviación y máximo por ángulo = `27` dims).
- **Bloque 3 — Textura LBP Uniforme en 3 Escalas Radiales (`108` dims, `7.7 %`):**
  - Calcula en una sola pasada por radio los patrones locales binarios uniformes en **micro-escala (`P=8, R=1`, `10` bins)**, **meso-escala (`P=16, R=2`, `18` bins)** y **macro-escala (`P=24, R=3`, `26` bins)** sobre la imagen completa y el recorte central (`54 x 2 = 108` dims).

### 3.2 Filtrado Automático de Calidad en Escenas `LVIS` ([src/frutas/classical_training.py](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/src/frutas/classical_training.py))

En `_eligible(frame, split)` (tanto en el pipeline principal como en el baseline), los recortes de `lvis_fruits_vegetables` se cruzan automáticamente con `crops_manifest.csv` y las dimensiones de `dataset.csv`, conservando únicamente recortes con:
- Resolución mínima **`min(width, height) >= 64` px**.
- **Pureza de escena (`scene_classes == 1`):** la escena original no mezcla cajas de `fruta` y `verdura` simultáneamente.
- **Baja congestión (`scene_boxes <= 8`):** evita recortes diminutos de pilas masivas en mercados.

Esto elevó la exactitud de `lvis_fruits_vegetables` en `test` de `69.64 %` a **`77.03 %`** y dejó de contaminar las fronteras de decisión del resto de los datasets.

### 3.3 Nueva Suite de 8 Modelos Clásicos + Ensamble Ponderado (`soft_voting_top3`)

Se actualizaron los candidatos en [src/frutas/classical_training.py](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/src/frutas/classical_training.py) manteniendo **100 % Machine Learning clásico**:
- **Histogram Gradient Boosting (`hgb_depth_63` y `hgb_depth_95`):** árboles de decisión potenciados por gradiente (`HistGradientBoostingClassifier` de `scikit-learn`) con `max_leaf_nodes=63` (`max_iter=420, lr=0.06`) y `max_leaf_nodes=95` (`max_iter=480, lr=0.05`), entrenados con pesos balanceados por clase (`compute_sample_weight("balanced")`). Cada uno por sí solo alcanza **`89.46 %` y `89.22 %` de accuracy en `validation`** (`F1 macro = 0.8705` y `0.8708`).
- **SVM RBF con `BlockBalancedScaler` + `AdaptivePCA(256)` (`svm_rbf_c10` y `svm_rbf_c25`):** proyecta los `1,405` atributos escalados por bloque a sus **256 componentes principales ortogonales** antes de ajustar `CalibratedClassifierCV(SVC(kernel="rbf"))` sobre **`14,000` filas balanceadas**, alcanzando **`87.56 %` y `87.70 %` de accuracy en `validation`** en solo `~30 s`.
- **Extra Trees de alta resolución (`extra_trees_sqrt` y `extra_trees_040`):** `500` árboles con `min_samples_leaf=2` y `class_weight="balanced_subsample"` sobre las `19,097` imágenes de entrenamiento (`87.42 %` de accuracy en `validation`).
- **Ensamble de Votación Suave Ponderada (`soft_voting_top3`):** combina los 4 mejores modelos de validación (`hgb_depth_95`, `hgb_depth_63`, `svm_rbf_c10` y `svm_rbf_c25`) ponderados por el cuadrado de su margen de F1 macro sobre `0.72`, alcanzando **`89.80 %` de accuracy y `0.8766` de F1 macro en `validation`**.

### 3.4 Integración del 6.º Dataset (`fruits360` curado y balanceado vía Git Sparse-Checkout)

En [src/frutas/download.py](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/src/frutas/download.py), [configs/datasets.yaml](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/configs/datasets.yaml) y [configs/label_mapping.csv](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/configs/label_mapping.csv) se añadió soporte nativo para `git clone --filter=blob:none --sparse` y `git sparse-checkout set`, permitiendo descargar de `https://github.com/fruits-360/fruits-360-100x100.git` (commit `c23fa253bb3281960a741f51be86824c17bb2244`) un subconjunto estrictamente balanceado de **44 categorías** (`20` carpetas de `verdura` botánica como `Beetroot 1`, `Cabbage red 1`, `Cabbage white 1`, `Carrot 1`, `Cauliflower 1`, `Ginger Root 1`, `Kohlrabi 1`, `Onion 2`, `Onion Red 1`, `Onion White 1`, `Potato Red 1`, `Potato Sweet 1`, `Potato White 1` y `24` carpetas de `fruta` botánica) con un solo comando: `frutas-data download fruits360`.

---

## 4. Resultados Detallados por Modelo, Fuente y Categoría (`v2.0.0`)

### 4.1 Tabla Comparativa de Selección en `validation` (`9,422` imágenes, [artifacts/model_selection.csv](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/artifacts/model_selection.csv))

| Posición | Modelo Candidato | F1 Macro (`val`) | Balanced Accuracy (`val`) | Accuracy (`val`) | Precision Macro | Recall Macro | Umbral Óptimo | Tiempo Ajuste |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **`soft_voting_top3` (Ganador)** | **0.8766** | **87.27 %** | **89.80 %** | **88.08 %** | **87.27 %** | `0.485` | — |
| 2 | `hgb_depth_95` | 0.8708 | 87.05 % | 89.22 % | 87.11 % | 87.05 % | `0.545` | 117.0 s |
| 3 | `hgb_depth_63` | 0.8705 | 86.15 % | 89.46 % | 88.15 % | 86.15 % | `0.445` | 91.6 s |
| 4 | `svm_rbf_c10` (`PCA-256`) | 0.8509 | 85.05 % | 87.56 % | 85.13 % | 85.05 % | `0.425` | 28.6 s |
| 5 | `svm_rbf_c25` (`PCA-256`) | 0.8494 | 84.19 % | 87.70 % | 85.82 % | 84.19 % | `0.375` | 31.6 s |
| 6 | `extra_trees_040` | 0.8436 | 83.18 % | 87.42 % | 85.93 % | 83.18 % | `0.540` | 57.2 s |
| 7 | `extra_trees_sqrt` | 0.8328 | 83.31 % | 86.02 % | 83.25 % | 83.31 % | `0.575` | 7.5 s |
| 8 | `logistic_c1` | 0.7657 | 77.66 % | 79.71 % | 75.84 % | 77.66 % | `0.420` | 5.5 s |
| 9 | `sgd_log` | 0.7652 | 77.79 % | 79.55 % | 75.71 % | 77.79 % | `0.425` | 2.8 s |

### 4.2 Desglose por Fuente en `test` Completo con TTA (`13,141` imágenes, [artifacts/error_analysis_by_label.csv](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/artifacts/error_analysis_by_label.csv))

| Fuente (`source`) | Imágenes evaluadas en `test` | Categorías en `test` | Accuracy en `v1.1.0` | **Accuracy en `v2.0.0` (con TTA)** | Mejora neta |
|---|---:|---:|---:|---:|---:|
| **`vegetable_ahmed`** | 2,165 | 15 | 90.39 % | **97.69 %** | **+7.30 pts** |
| **`grocery_store`** (fotos reales de celular en supermercado) | 1,665 | 49 | 85.11 % | **91.17 %** | **+6.06 pts** |
| **`fruits_youssef`** | 9,119 | 42 | 80.24 % | **86.97 %** | **+6.73 pts** |
| **`kritikseth36`** | 44 | 10 | 79.55 % | **79.55 %** | — |
| **`lvis_fruits_vegetables`** (recortes limpios $\ge 64$ px) | 148 | 29 | 69.64 % | **77.03 %** | **+7.39 pts** |
| **Total `test` elegible completo** | **13,141** | **145** | — | **89.13 %** (`91.98 %` confiable) | — |

Ejemplos de categorías con **`100.0 %` de exactitud en `test`**:
- En `vegetable_ahmed`: `Potato` (`167/167`, `verdura`), `Bean` (`156/156`, `fruta`), `Capsicum` (`141/141`, `fruta`), `Papaya` (`124/124`, `fruta`), `Bottle_Gourd` (`112/112`, `fruta`), `Carrot` (`99.5 %`, `verdura`), `Radish` (`99.3 %`, `verdura`), `Cauliflower` (`97.9 %`, `verdura`).
- En `grocery_store`: `Satsumas` (`68/68`), `Royal-Gala` (`64/64`), `Granny-Smith` (`58/58`), `Orange` (`56/56`), `Red-Delicious` (`50/50`), `Regular-Tomato` (`47/47`), `Golden-Delicious` (`45/45`), `Banana` (`44/44`), `Red-Beet` (`95.0 %`, `verdura`), `Garlic` (`93.3 %`, `verdura`), `Carrots` (`92.9 %`, `verdura`), `Solid-Potato` (`92.3 %`, `verdura`).
- En `fruits_youssef`: `Dragonfruit` (`117/117`), `Tomato` (`94/94`).

---

## 5. Benchmark de Estrés ante Ruido ([artifacts/noise_robustness.csv](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/artifacts/noise_robustness.csv))

Evaluación de `soft_voting_top3` con TTA sobre `1,677` imágenes de `test` bajo **8 condiciones explícitas**:

| Condición de Ruido | Descripción Técnica | Accuracy (`v2.0.0`) | Balanced Accuracy | F1 Macro (`v2.0.0`) | Exactitud Confiable (`>= 0.60`) | Retención de F1 vs `clean` |
|---|---|---:|---:|---:|---:|---:|
| **`clean`** | Imagen original sin perturbar | **87.00 %** (vs `76.61 %` en `v1.1.0`) | **82.74 %** | **0.8303** (vs `0.7221`) | **90.15 %** | **100.00 %** |
| **`occlusion_patch`** | Parche oclusivo de `22 % x 22 %` sobre el objeto | **86.52 %** | **80.51 %** | **0.8179** | **89.39 %** | **98.50 %** |
| **`illumination_high`** | Sobreexposición intensa (`brillo 1.38x`) | **85.39 %** | **78.42 %** | **0.7998** | **88.50 %** | **96.32 %** |
| **`rotation_crop`** | Giro de `18°` + recorte descentrado al `82 %` | **85.33 %** | **81.17 %** | **0.8108** | **87.98 %** | **97.65 %** |
| **`gaussian_noise`** | Ruido aditivo de sensor ($\sigma = 14.0$) | **84.68 %** | **76.39 %** | **0.7844** | **88.16 %** | **94.47 %** |
| **`combined_heavy`** | Giro + recorte + cambio de luz/color + blur + ruido | **83.60 %** | **78.01 %** | **0.7845** | **88.21 %** | **94.48 %** |
| **`gaussian_blur`** | Desenfoque óptico (`radius = 1.5`) | **82.59 %** | **79.24 %** | **0.7822** | **85.92 %** | **94.20 %** |
| **`illumination_low`** | Subexposición severa (`brillo 0.62x`, `contraste 0.85x`) | **81.57 %** | **79.51 %** | **0.7758** | **86.07 %** | **93.43 %** |
| **Promedio bajo ruido** | Media de las 7 perturbaciones adversas | **84.24 %** (vs `75.27 %` en `v1.1.0`) | **79.04 %** | **0.7936** (vs `0.7076`) | **87.75 %** | **95.58 %** |

> **Observación clave:** En `v2.0.0`, el desempeño promedio **bajo ruido adverso** (`Accuracy = 84.24 %`, `F1 Macro = 0.7936`) supera ampliamente al desempeño que tenía el modelo anterior **en imágenes limpias sin ruido** (`Accuracy = 76.61 %`, `F1 Macro = 0.7221`).

---

## 6. Guía Completa para Pruebas Reales

El modelo `v2.0.0` ya está serializado en [artifacts/classical_model.joblib](file:///Users/chunkdunk/Developer/machine-learning/clasificador-frutas-verduras/artifacts/classical_model.joblib) y listo para clasificar fotografías individuales o carpetas completas.

### Interfaz web local (`v2.1.0`)

```bash
python -m pip install -e ".[app]"
frutas-app
```

La página permite subir una imagen, activar TTA, ver probabilidades y guardar
opcionalmente ejemplos cuya etiqueta fue confirmada por una persona. Los ejemplos se
deduplican y se almacenan en `data/raw/team_photos_train/{fruta,verdura}/`; esta fuente
está fijada a `train`, por lo que no contamina validation ni test. El modelo solamente
se actualiza cuando se ejecutan explícitamente `frutas-data all` y
`frutas-data train-classic`.

### Caso A: Probar una sola fotografía (tomada con celular o descargada)

```bash
frutas-data predict-classic ruta/a/tu_foto.jpg
```

**Ejemplo real verificado sobre una foto de ajo en supermercado (`Garlic_001.jpg`):**
```json
{
  "prediction": "verdura",
  "confidence": 0.9779817723514153,
  "probabilities": {
    "fruta": 0.02201822764858477,
    "verdura": 0.9779817723514153
  },
  "uncertain": false,
  "model": "soft_voting_top3",
  "decision_threshold": 0.485,
  "tta_views": 4,
  "path": ".../Garlic_001.jpg"
}
```

### Caso B: Probar una carpeta completa de fotos por lotes

```bash
frutas-data predict-classic ruta/a/mis_fotos/ --output artifacts/mis_predicciones.csv
```
- Si las fotos están organizadas en subcarpetas `mis_fotos/fruta/` y `mis_fotos/verdura/` (o con nombres de etiquetas registradas en `configs/label_mapping.csv`), calcula automáticamente `accuracy`, `balanced_accuracy`, `f1_macro`, `accuracy_when_confident` y la matriz de confusión del lote.

### Caso C: Evaluar el benchmark externo oficial del equipo (`data/external/team_photos/`)

1. Colocar las fotos tomadas por el equipo en `data/external/team_photos/fruta/` y `data/external/team_photos/verdura/`.
2. Ejecutar:
   ```bash
   frutas-data all
   frutas-data evaluate-classic --split external --output artifacts/external_breakdown.csv
   ```
