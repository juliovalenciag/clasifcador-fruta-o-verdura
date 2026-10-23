# Clasificador binario de frutas y verduras (v2.2.0)

Proyecto de Machine Learning clásico que recibe una imagen (o una carpeta completa de fotografías) y responde únicamente una de dos clases bajo **criterio botánico estricto**: **fruta** o **verdura**. Las especies y variedades se conservan exclusivamente como metadatos de auditoría y análisis de errores.

- **Fruta (criterio botánico):** estructura desarrollada a partir del ovario de la flor (incluye manzana, plátano, naranja, tomate, pepino, berenjena, pimiento, chile, calabaza, calabacita, maíz y vainas de leguminosas).
- **Verdura (criterio botánico):** raíz, tubérculo, bulbo, rizoma, tallo, hoja o flor comestible (incluye zanahoria, papa, camote, betabel, rábano, cebolla, ajo, puerro, jengibre, espárrago, apio, lechuga, espinaca, col, brócoli, coliflor y alcachofa).
- **Excluidos del alcance binario (`rejected`):** hongos (`mushroom`), semillas secas (`almond`, `chickpea`, `peanuts`) y alimentos procesados o empaquetados (`pickle`, `tofu`, jugos, lácteos).

La solución es **100 % Machine Learning clásico** (sin redes neuronales). En `v2.0.0`, cada imagen se transforma en un vector determinista de **1,405 características** (`float32`) balanceado por familias e invarianzas físicas:
- **Color, cromaticidad, índices botánicos y separación figura-fondo (839 dims — 59.7 %):** histogramas `1D` (`16` bins) de **HSV** y **CIE Lab** (global + centro), histogramas (`12` bins) de **4 índices botánicos/radiométricos** ($ExG = 2G - R - B$, $R-G$, $Y-B$, croma CIE $C^*$), histogramas de matiz (`Hue`) ponderados por saturación (suprimen fondos blancos o grises), histogramas conjuntos `2D` de cromaticidad `(H, S)` (`8x4`) y `(a*, b*)` (`6x6`), perfiles en **3 anillos radiales concéntricos** con contraste interior-borde, **segmentación automática figura-fondo** en espacio `Lab + Croma` (con momentos geométricos de excentricidad, compacidad y aspecto) y cuadrícula espacial `3x3`.
- **Forma, energía de bordes y HOG multiescala agrupado (458 dims — 32.6 %):** energía de gradiente **Sobel** y alta frecuencia **Laplaciana** en anillos concéntricos y cuadrícula `4x4` (`44` dims) + **HOG multiescala** agrupado en `3x3` macro-regiones (media y máximo = `162` dims), celdas centrales `5x5` del objeto (`225` dims) y resumen global invariante a traslación (`27` dims).
- **Textura LBP multiescala (108 dims — 7.7 %):** histogramas **LBP uniforme** en **3 escalas radiales** (`P=8, R=1` micro-textura; `P=16, R=2` meso-textura; `P=24, R=3` macro-textura) tanto globales como centrales.
- **Suite de 8 modelos clásicos + ensamble ponderado:** **Histogram Gradient Boosting** (`max_leaf_nodes` 63 y 95), **SVM RBF con `AdaptivePCA(256)` y `BlockBalancedScaler`** (`14,000` filas balanceadas), **Extra Trees** y clasificadores lineales. Desde `v2.2.0` se entrena con **todo train** (`78,719` imágenes en vez de `19,097`) leyendo una caché de características; el ganador en validation es `hgb_depth_95`.
- **Filtrado automático de calidad en escenas complejas (`LVIS`):** descarta parches diminutos (`< 64x64` px) y recortes provenientes de escenas mixtas (`fruta + verdura` simultáneas) o saturadas (`> 8` objetos por imagen).
- **Inferencia robusta (TTA):** promedio de **4 vistas en tiempo de prueba** (original, espejo horizontal, recorte central `86 %` y normalización de contraste `autocontrast`).

---

## Trabajo en equipo: descarga solo lo que tu tarea necesita

El modelo y las características se comparten en **Hugging Face Hub** (repos privados
del equipo definidos en [configs/hub.yaml](configs/hub.yaml)). Solo una persona —quien
agrega datasets o cambia el extractor— necesita los ~10 GB de imágenes.

| Quiero… | Necesito | Tamaño | Comandos |
|---|---|---:|---|
| Probar el modelo, usar la app, clasificar mis fotos | el modelo | ~40 MB | `frutas-data hub pull-model` → `frutas-app` |
| Entrenar, ajustar hiperparámetros, comparar modelos | caché de características | ~530 MB | `frutas-data hub pull-features` → `frutas-data train-classic --features` |
| Entrenar sin usar mi computadora | nada local | — | [notebooks/03_entrenar_en_la_nube.ipynb](notebooks/03_entrenar_en_la_nube.ipynb) en Google Colab |
| Agregar un dataset o fotos, cambiar `classic_features.py` | imágenes originales | ~10 GB | sección *Pipeline completo* ↓ |

Configuración única por integrante (el token nunca va al repositorio):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks,app]"
hf auth login          # token de https://huggingface.co/settings/tokens
```

Flujo de la persona responsable de datos cuando cambian las imágenes:

```bash
frutas-data all                      # inventario, duplicados y splits
frutas-data extract-features         # incremental: solo extrae SHA-256 nuevos (~2 min completo)
frutas-data hub push-features        # el resto del equipo hace pull-features
frutas-data train-classic --features # entrena desde la caché, sin releer imágenes
frutas-data hub push-model --tag v2.2.0
```

Para que todos usen exactamente el mismo modelo, fije la etiqueta en
`model_revision` de `configs/hub.yaml`. La carpeta [deploy/hf_space](deploy/hf_space)
contiene lo necesario para publicar la interfaz como un Space de Hugging Face
(demo web sin instalar nada).

> Los modelos son archivos `joblib` (pickle): cárguelos solo desde los repositorios del
> equipo y con la versión fijada de scikit-learn (`1.9.1`).

## Pipeline completo (requiere las imágenes)

```bash
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

# Caché de características (una vez) y entrenamiento del modelo clásico
frutas-data extract-features
frutas-data train-classic --features
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
Devuelve en JSON la clase predicha (`fruta` o `verdura`), la confianza (`0.0` a `1.0`), las probabilidades por clase, el umbral de decisión calibrado en validation (`0.56` en `v2.2.0`), el indicador `"uncertain": true/false` (si la confianza es menor a `0.60`) y las `4` vistas TTA utilizadas.

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

## Estado actual verificado (`v2.2.0`)

Mismas particiones, mismas imágenes de `test` y mismo extractor (`1,405` variables) que
`v2.0.0`; lo que cambió es cuántos datos ve el modelo y cómo se seleccionan.

| Métrica | `v2.0.0` | **`v2.2.0`** |
|---|---:|---:|
| Imágenes de train usadas | `19,097` (tope 180 por fuente/etiqueta) | **`78,719`** (todas) |
| Modelo seleccionado en validation | `soft_voting_top3` (4 modelos, 43 MB) | **`hgb_depth_95`** (1 modelo, 8.7 MB) |
| F1 macro en validation (`9,422`) | `0.8766` | **`0.9019`** |
| Test muestreado (`9,293`) — accuracy | `89.65 %` | **`91.03 %`** |
| Test muestreado — balanced accuracy | `87.06 %` | **`88.34 %`** |
| Test muestreado — F1 macro | `0.8701` | **`0.8863`** |
| Precision / recall `verdura` | `80.95 %` / `81.34 %` | **`84.45 %` / `82.40 %`** |
| Precision / recall `fruta` | `92.95 %` / `92.78 %` | **`93.42 %` / `94.28 %`** |
| Accuracy confiable (`>= 0.60`) / cobertura | `91.97 %` / `92.60 %` | **`92.43 %` / `96.04 %`** |
| Test completo con TTA (`13,141`) — accuracy / F1 | `89.13 %` / `0.8523` | **`90.50 %` / `0.8701`** |
| Ruido (`1,677` imgs) — F1 limpio / F1 medio con ruido | `0.8303` / `0.7936` | `0.8367` / `0.7927` |

Exactitud por fuente en `test` completo con TTA:

| Fuente | `v2.0.0` | `v2.2.0` |
|---|---:|---:|
| `vegetable_ahmed` (`2,165`) | `97.69 %` | **`98.71 %`** |
| `fruits_youssef` (`9,119`) | `86.97 %` | **`88.85 %`** |
| `lvis_fruits_vegetables` (`148`) | `77.03 %` | **`78.38 %`** |
| `kritikseth36` (`44`) | `79.55 %` | `79.55 %` |
| `grocery_store` (`1,665`, fotos reales) | **`91.17 %`** | `90.21 %` |

Limitaciones conocidas:

- `grocery_store` (la única fuente de fotos reales de celular) baja `~1` pt: al usar todo
  train, `fruits_youssef` pesa más. Sus verduras siguen siendo el punto débil
  (`Cabbage` 0/19, `Asparagus` 1/14, `Leek` 9/21).
- `data/external/team_photos/` está vacío: aún no existe una medición en fotos propias
  del equipo, que es la que mejor representa el uso real.
- Experimentos descartados (validation, mismas particiones): extraer a `128 px`
  (F1 `0.9016` vs `0.9011`, el doble de tiempo) y añadir Bag of Visual Words con
  DAISY + k-means (`+0.003` F1). El descriptor está cerca de su techo; las siguientes
  mejoras deben venir de la calidad de los datos (sección 9.5 de la guía).

Artefactos locales en `artifacts/` (compartibles con `frutas-data hub push-model`):
`classical_model.joblib`, `classical_metrics.json`, `model_selection.csv`,
`error_analysis_by_label.csv` y `noise_robustness.csv`. El modelo `v2.0.0` anterior
queda respaldado en `artifacts/backup_v2.0.0/`.

La guía operativa detallada está en [guia-equipo.md](guia-equipo.md) y el informe técnico completo en [reporte-estado-proyecto.md](reporte-estado-proyecto.md).
