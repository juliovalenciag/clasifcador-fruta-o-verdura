# Guía del equipo: clasificador fruta o verdura

## 1. Alcance oficial

El sistema recibe una imagen y produce exactamente una predicción binaria:

```text
imagen -> fruta | verdura
```

No debe identificar la especie ni la variedad. “Manzana”, “plátano”, “papa” o
“Granny Smith” pueden conservarse como etiquetas originales para analizar la
diversidad del dataset, pero nunca serán la variable objetivo.

El objetivo técnico es que el clasificador mantenga buen desempeño con ruido real:
fondos diferentes, cambios de iluminación, rotación, recortes, desenfoque,
compresión, objetos parcialmente tapados y frutas o verduras cortadas.

## 2. Criterio oficial de etiquetado

El proyecto adopta el **criterio botánico**. La regla se aplica de forma uniforme a todas las fuentes:

- **Fruta:** estructura que se desarrolla del ovario de la flor. Incluye tomate,
  pepino, berenjena, calabaza, chile, pimiento, maíz y vainas de leguminosas, aunque
  culinariamente algunas se llamen verduras.
- **Verdura:** parte vegetal distinta del fruto, como raíz, tubérculo, bulbo, tallo,
  hoja o flor. Incluye zanahoria, papa, cebolla, ajo, jengibre, lechuga, espinaca,
  col y coliflor.

Las etiquetas generales `peas`, `soy beans`, `corn` y `sweetcorn` del dataset
principal se asignan a fruta. Esta convención queda documentada aunque una imagen
concreta muestre solamente la semilla y no la vaina completa.

Hongos, comida preparada, ensaladas mixtas e imágenes sin una fruta o verdura clara
se excluyen; este proyecto no tiene una tercera clase. `status=review` se reserva para
errores de fuente o casos nuevos que no encajen claramente en la regla, no para volver
a mezclar criterios culinarios y botánicos.

## 3. Qué se comparte y dónde

### GitHub

Se versionan:

- Código fuente y pruebas.
- Notebooks sin datos incrustados.
- `configs/datasets.yaml`.
- `configs/classes.csv`.
- `configs/label_mapping.csv`.
- Documentación (`README.md`, `guia-equipo.md`, `reporte-estado-proyecto.md`).

No se versionan:

- Imágenes.
- Archivos ZIP de datasets.
- Características numéricas ni manifiestos generados.
- Modelos entrenados (`artifacts/`).
- Tokens de Kaggle o GitHub.

### Google Drive compartido

Drive se utiliza exclusivamente para imágenes y datasets. No se guardan ahí código,
modelos, métricas, manifiestos ni documentación. Carpeta raíz del equipo:

`https://drive.google.com/drive/folders/1jT810RWNQYZ5WQAqMPHE1ISlPlIxCLjV`

```text
frutas-verduras/
├── datasets_publicos/          # opcional; originales sin modificar
├── team_photos_train/
│   ├── fruta/
│   └── verdura/
├── team_photos_external/
│   ├── fruta/
│   └── verdura/
```

Los datasets públicos reproducibles se descargan desde su fuente. No es necesario
duplicarlos en Drive, salvo que la fuente deje de estar disponible.

La carpeta raíz actualmente permite editar a cualquier persona con el enlace. Antes
de distribuirlo, el propietario debe abrir **Compartir -> Acceso general**, cambiarlo
a **Restringido** y agregar por correo a cada integrante como editor. Si se conserva
“cualquier persona con el enlace”, se recomienda dejarlo como lector, no editor.

Flujo para las fotos del equipo:

1. Cada imagen se sube en una sola carpeta `fruta/` o `verdura/`.
2. Se nombra `integrante_fecha_secuencia.ext`, por ejemplo
   `ana_20261015_003.jpg`.
3. No se suben ráfagas casi idénticas ni imágenes descargadas de Internet.
4. Se descarga `team_photos_train` y se copia a
   `data/raw/team_photos_train/`.
5. Se descarga `team_photos_external` y se copia a
   `data/external/team_photos/`.
6. Se ejecuta `frutas-data all` para validar, deduplicar y registrar las fotos.

`team_photos_external` se congela antes de ajustar modelos. Ninguna foto de esa
carpeta se mueve después a train y sus resultados no se usan para elegir
hiperparámetros.

Cada integrante utiliza sus propias credenciales de Kaggle. Ningún token se comparte
ni se escribe en el repositorio.

### Configuración personal de Kaggle

Cada integrante debe:

1. Crear o utilizar su cuenta de Kaggle.
2. Abrir la página del dataset y aceptar sus condiciones cuando Kaggle lo solicite.
3. Crear un API token desde `https://www.kaggle.com/settings/api`.
4. Configurarlo sin guardarlo dentro del proyecto.

En macOS o Linux, para la sesión actual:

```bash
export KAGGLE_API_TOKEN="TOKEN_PERSONAL"
```

En PowerShell:

```powershell
$env:KAGGLE_API_TOKEN="TOKEN_PERSONAL"
```

En Google Colab se crea un secreto llamado `KAGGLE_API_TOKEN` y se habilita el
acceso del notebook. Después se carga sin imprimirlo:

```python
import os
from google.colab import userdata

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
```

El token nunca se pega en una celda, un archivo `.env`, una captura o un commit.

## 4. Estructura local

```text
clasificador-frutas-verduras/
├── configs/
│   ├── classes.csv
│   ├── datasets.yaml
│   └── label_mapping.csv
├── data/
│   ├── raw/                 # originales inmutables
│   ├── derived/             # recortes reproducibles (ej. LVIS YOLO)
│   ├── interim/             # manifiestos y auditorías
│   ├── processed/           # índice final con splits
│   ├── external/            # prueba externa congelada
│   └── quarantine/          # imágenes que requieren revisión
├── notebooks/
├── src/frutas/
│   ├── app.py              # interfaz web Streamlit
│   └── feedback.py         # validación y guardado supervisado
├── tests/
└── artifacts/               # modelos y métricas locales
```

No se renombran, convierten o sobrescriben originales dentro de `data/raw/`.

## 5. Fuentes de datos

Las fuentes conocidas están registradas en `configs/datasets.yaml`:

- `kritikseth36`: Fruit and Vegetable Image Recognition (36 etiquetas originales, activo).
- `fruits_youssef`: Fruit and Vegetables Classification (50 categorías, versión 5, activo).
- `lvis_fruits_vegetables`: LVIS Fruits And Vegetables (61 etiquetas canónicas de detección YOLO convertidas a recortes en `data/derived/`, activo).
- `grocery_store`: Grocery Store Dataset de Klasson et al. (`5,421` fotografías reales de teléfono en supermercados, 81 clases finas; 49 clases botánicas aprobadas y 32 clases de empaquetados/hongos rechazadas, activo).
- `vegetable_ahmed`: Vegetable Image Dataset de Ahmed et al. (`21,000` imágenes en 15 categorías: 6 verduras botánicas y 9 frutos culinarios; activo, descargado y mapeado al 100 % en `configs/label_mapping.csv`).
- `fruits360`: Fruits-360 (`https://github.com/fruits-360/fruits-360-100x100.git`, commit fijado `c23fa253bb3281960a741f51be86824c17bb2244`, configurado con `git sparse-checkout` para un subconjunto estrictamente balanceado de **44 categorías**: 20 carpetas de `verdura` botánica como `Carrot 1`, `Cauliflower 1`, `Cabbage white 1`, `Onion 2`, `Potato Red 1`, `Ginger Root 1`, `Beetroot 1`, `Kohlrabi 1` y 24 carpetas de `fruta` botánica).
- `fruits262`: Fruits-262 (inactivo por contener solo frutas).
- `google_shared`: Fuente compartida por Google (opcional).
- `team_photos_train` y `team_photos`: Fotografías del equipo (entrenamiento y prueba externa).

Una fuente que contiene exclusivamente frutas no debe añadirse sin verduras comparables.
De lo contrario, el modelo puede aprender el estilo de la fuente en vez de la diferencia
fruta/verdura.

## 6. Preparación del entorno

```bash
cd clasificador-frutas-verduras
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,app]"
pytest
```

En Windows, la activación normalmente es:

```powershell
.venv\Scripts\Activate.ps1
```

La instalación se realiza desde `pyproject.toml`; no se deben instalar versiones al
azar en cada computadora ni mantener listas distintas por integrante. En particular,
`scikit-learn==1.9.1` está fijado porque los modelos `joblib` no garantizan compatibilidad
entre versiones distintas de scikit-learn.

## 7. Colocación de datasets

Cada dataset conserva su estructura original en la ruta indicada por su entrada de
`configs/datasets.yaml`:

```text
data/raw/kritikseth36/
data/raw/fruits_youssef/
data/raw/lvis_fruits_vegetables/
data/derived/lvis_fruits_vegetables/
data/raw/grocery_store/
data/raw/vegetable_ahmed/
data/raw/fruits360/
data/raw/fruits262/
data/raw/google_shared/
```

Los datasets públicos activos se descargan y preparan con:

```bash
frutas-data download kritikseth36
frutas-data download fruits_youssef
frutas-data download lvis_fruits_vegetables
frutas-data prepare-lvis
frutas-data download grocery_store
frutas-data download vegetable_ahmed
# Opcional: 6.ª fuente curada con git sparse-checkout (44 clases balanceadas)
frutas-data download fruits360
```

La configuración fija la versión o commit de cada dataset (`kritikseth36`, `fruits_youssef`,
`lvis_fruits_vegetables`, `vegetable_ahmed`, `fruits360`) o descarga sus particiones oficiales
(`grocery_store` vía Git).
Una descarga completa se reutiliza; `--force` solamente se emplea cuando se verificó que
la descarga local quedó incompleta y se desea reemplazarla.

Las fotografías del equipo se separan desde el principio:

```text
data/raw/team_photos_train/fruta/
data/raw/team_photos_train/verdura/
data/external/team_photos/fruta/
data/external/team_photos/verdura/
```

La carpeta externa debe congelarse antes de entrenar. Sus imágenes no se copian a
train después de ver los resultados.

## 8. Mapeo binario de etiquetas

Cada nombre de carpeta original se traduce explícitamente en
`configs/label_mapping.csv`:

```csv
source,original_label,class_id,status,notes
kritikseth36,apple,fruta,approved,
kritikseth36,carrot,verdura,approved,
kritikseth36,tomato,fruta,approved,Criterio botánico
```

Estados válidos del proceso de revisión:

- `approved`: participa en el entrenamiento.
- `review`: requiere una decisión humana.
- `rejected`: se excluye por etiqueta incorrecta o fuera de alcance.
- `unmapped`: el pipeline encontró una etiqueta todavía no registrada.

No se permite inferir automáticamente que todas las carpetas con cierto nombre son
correctas. Cada fuente puede utilizar convenciones diferentes.

## 9. Pipeline reproducible de datos

El flujo completo es:

```text
originales
   -> inventario y validación
   -> mapeo a fruta/verdura
   -> duplicados exactos y perceptuales
   -> splits agrupados
   -> auditoría
   -> entrenamiento
```

Puede ejecutarse de una vez:

```bash
frutas-data all
```

O etapa por etapa:

```bash
frutas-data inventory
frutas-data dedupe --phash-distance 4
frutas-data split --train 0.70 --validation 0.15 --test 0.15 --seed 42
frutas-data audit
```

### 9.1 Inventario

`inventory` genera `data/interim/manifest.csv` con:

- Fuente y ruta.
- Etiqueta original.
- Clase binaria aprobada.
- Dimensiones, formato y tamaño.
- Validez de la imagen.
- SHA-256 y hash perceptual (`pHash`).
- Split original, cuando existe.

Una imagen corrupta se registra, pero no detiene todo el inventario.

### 9.2 Duplicados

`dedupe` no borra archivos. Agrupa:

- Copias exactas mediante SHA-256.
- Copias visualmente parecidas mediante hash perceptual (`pHash`, distancia de Hamming <= 4) indexado por segmentos de bits.

Si un mismo contenido aparece etiquetado como fruta y verdura, el grupo se marca
como conflicto y se excluye hasta su revisión.

### 9.3 Particiones

`split` crea `data/processed/dataset.csv`. Todos los miembros de un grupo duplicado
permanecen en una sola partición, evitando que una copia aparezca en entrenamiento y
otra en prueba.

La distribución predeterminada es:

- 70 % entrenamiento.
- 15 % validación.
- 15 % prueba interna.
- Fotografías con `role=external` quedan en prueba externa.

Las particiones preservan asignaciones previas y respetan `original_split` cuando la
fuente publica particiones oficiales, unificando cualquier grupo duplicado por mayoría
determinista.

### 9.4 Auditoría

`audit` genera `data/interim/data_report.md`. Antes de entrenar se debe verificar:

- Cero fugas de grupos duplicados entre splits.
- Cero conflictos de etiqueta dentro de splits activos.
- Todas las etiquetas utilizadas con estado `approved`.
- Ambas clases presentes en train, validation y test.
- Balance razonable entre fruta y verdura.
- Diversidad de fuentes, especies, fondos y condiciones.

## 10. Reglas para fotografías del equipo

Cada clase debe incluir ejemplos de:

- Objetos enteros y cortados.
- Fondos claros, oscuros y con textura.
- Luz natural, artificial, intensa y tenue.
- Distancias y ángulos distintos.
- Oclusión parcial.
- Diferentes cámaras o teléfonos.
- Una pieza y varias piezas del mismo tipo.

Evitar:

- Diez fotografías casi idénticas de una sola sesión.
- Ensaladas, mercados o cajas que mezclen ambas clases.
- Productos procesados o platillos preparados.
- Dibujos y renders, salvo que formen parte explícita de la evaluación.
- Marcas de agua dominantes.
- Imágenes cuya etiqueta sea discutible.

Las ráfagas o variantes de una misma fotografía deben mantenerse en el mismo grupo.

## 11. Ruido sintético

Los aumentos se aplican únicamente durante entrenamiento. El conjunto de validación
y los tests conservan sus imágenes originales.

El entrenamiento clásico implementado utiliza variaciones reproducibles de:

- Recorte y cambio de escala.
- Rotación moderada e inversión horizontal.
- Cambio de brillo, contraste y saturación.
- Desenfoque gaussiano.
- Ruido de sensor.

Estas variantes se generan en memoria y no duplican archivos dentro de `data/raw/`.
Los aumentos no sustituyen fotografías reales con fondos variados.

## 12. Extracción de características y baseline clásico (`v2.0.0`)

Cada imagen se redimensiona a `64x64` y se transforma en un vector numérico
determinista de **1,405 características** (`float32`), equilibrado para que ni
los árboles ni las máquinas de soporte vectorial queden dominados por celdas HOG
sensibles a traslación:

- **Bloque de Color, Cromaticidad, Índices Botánicos y Separación Figura-Fondo (`839` dimensiones — `59.7 %`):**
  - Histogramas `1D` (`16` bins) en los espacios **HSV** y **CIE Lab** (globales y del recorte central del `50 %`).
  - Histogramas (`12` bins) de **4 índices botánicos y radiométricos**: Excess Green ($ExG = 2G - R - B$), oposición Rojo-Verde ($R - G$), oposición Amarillo-Azul ($Y - B$) y Croma perceptual CIE $C^* = \sqrt{(a^*)^2 + (b^*)^2}$.
  - Histogramas de matiz (`Hue`, `16` bins) ponderados por saturación (`S * V`) que anulan el fondo blanco, gris o negro.
  - Histogramas conjuntos `2D` de cromaticidad `(H, S)` (`8x4 = 32` bins) y `(a*, b*)` (`6x6 = 36` bins) tanto globales como centrales.
  - Perfiles en **3 anillos radiales concéntricos** (núcleo interior, anillo medio y borde exterior) y vectores explícitos de contraste `interior - exterior` e `interior - medio` sobre los 13 canales.
  - **Segmentación automática figura-fondo** en espacio `(L*, a*, b*, S, C*)` comparando cada píxel contra el marco perimetral de la imagen, extrayendo media, desviación estándar y contraste del objeto más 6 descriptores geométricos de silueta (fracción de área, centralidad, ancho/alto de caja, razón de aspecto, excentricidad por autovalores y compacidad).
  - Momentos espaciales en cuadrícula `3x3` sobre canales de color e índices.
- **Bloque de Forma, Energía de Bordes y HOG Multiescala (`458` dimensiones — `32.6 %`):**
  - Magnitud de gradiente **Sobel** y energía de alta frecuencia **Laplaciana** en los 3 anillos concéntricos y en una cuadrícula `4x4` (`44` dims), diferenciando cáscaras lisas (frutas como manzana, tomate, pimiento) de cabezas florales y hojas rugosas (brócoli, coliflor, lechuga).
  - **HOG multiescala agrupado (`414` dims):** a partir de las celdas `8x8` (`7x7x9`), se extraen medias y máximos en `3x3` macro-regiones (`162` dims), las `5x5` celdas centrales del objeto (`225` dims) y el resumen global de orientación invariante a posición (`27` dims).
- **Bloque de Textura LBP Multiescala (`108` dimensiones — `7.7 %`):**
  - Histogramas **LBP uniforme** en **3 escalas radiales** (`P=8, R=1` micro-textura = `10` bins; `P=16, R=2` meso-textura = `18` bins; `P=24, R=3` macro-textura = `26` bins) sobre la imagen completa y el recorte central (`54 x 2 = 108` dims).

Los clasificadores lineales utilizan `BlockBalancedScaler(color_dim=839, lbp_dim=108)` y los modelos SVM RBF añaden `AdaptivePCA(max_components=256)` para proyectar el espacio equilibrado en sus 256 componentes ortogonales de mayor varianza antes del kernel radial.

Para ejecutar el baseline lineal rápido (`accuracy = 74.45 %`, `F1 macro = 0.7445` en test equilibrado):

```bash
frutas-data train-baseline --image-size 64 --max-per-class 1000
frutas-data predict-baseline ruta/a/imagen.jpg
```

## 13. Selección y ensamble clásico (`v2.0.0`)

El entrenamiento final se ejecuta con:

```bash
frutas-data train-classic \
  --image-size 64 \
  --augmentations 0 \
  --max-per-source-label 180 \
  --svm-max-rows 14000 \
  --seed 42
```

Además, antes de muestrear, `_eligible` aplica un **filtro automático de calidad sobre `lvis_fruits_vegetables`** usando `crops_manifest.csv`: se conservan únicamente recortes con resolución mínima de `64x64` píxeles procedentes de escenas de clase pura (`scene_classes == 1`) y no saturadas (`scene_boxes <= 8`), eliminando más de `22,000` parches contaminados donde cajas de fruta y verdura se traslapaban en ensaladas o platos mixtos.

El flujo evita escoger un modelo mirando los resultados de prueba:

```text
train (19,097 imgs) -> extracción 1,405 vars (v4) -> 8 candidatos clásicos + ensamble
validation (9,422 imgs) -> selección por F1 macro y calibración de umbral
train + validation (28,519 imgs) -> reentrenamiento del ganador (soft_voting_top3)
test (9,293 muestreadas / 13,141 elegibles) -> evaluación interna una sola vez
external -> evaluación separada, si existen fotografías externas
```

Los 8 candidatos individuales son:
1. `logistic_c1`: Regresión logística (`C=1.0`, `class_weight="balanced"`).
2. `sgd_log`: SGD con pérdida logística (`alpha=1e-4`, `class_weight="balanced"`).
3. `svm_rbf_c10`: `BlockBalancedScaler` + `AdaptivePCA(256)` + `CalibratedClassifierCV(SVC(C=10.0, kernel="rbf"))` sobre `14,000` filas balanceadas.
4. `svm_rbf_c25`: `BlockBalancedScaler` + `AdaptivePCA(256)` + `CalibratedClassifierCV(SVC(C=25.0, kernel="rbf"))` sobre `14,000` filas balanceadas.
5. `extra_trees_sqrt`: `ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample")`.
6. `extra_trees_040`: `ExtraTreesClassifier(n_estimators=500, max_features=0.35, min_samples_leaf=2, class_weight="balanced_subsample")`.
7. `hgb_depth_63`: **Histogram Gradient Boosting** (`HistGradientBoostingClassifier(max_iter=420, learning_rate=0.06, max_leaf_nodes=63)` con pesos balanceados por muestra).
8. `hgb_depth_95`: **Histogram Gradient Boosting** (`HistGradientBoostingClassifier(max_iter=480, learning_rate=0.05, max_leaf_nodes=95)` con pesos balanceados por muestra).

Finalmente, `soft_voting_top3` construye un ensamble de votación suave ponderada con los 4 mejores candidatos en validación (`hgb_depth_95`, `hgb_depth_63`, `svm_rbf_c10` y `svm_rbf_c25`), ponderando cada modelo según el cuadrado de su margen de F1 macro sobre `0.72`.

La selección prioriza F1 macro, con balanced accuracy y accuracy como desempates. Para
cada candidato también se busca, **solo en `validation`**, un umbral de decisión para la
probabilidad de `fruta` entre `0.30` y `0.70` (paso `0.005`). El ganador (`soft_voting_top3`) seleccionó `0.485`. El umbral de incertidumbre (`0.60`) es
independiente: decide cuándo marcar una salida como dudosa, no qué clase gana. El comando
crea:

- `artifacts/classical_model.joblib`: modelo y configuración de características.
- `artifacts/classical_metrics.json`: métricas completas y matriz de confusión.
- `artifacts/model_selection.csv`: tabla ordenada de candidatos.

## 13.1 Guía de uso para pruebas reales e inferencia a prueba de ruido

El clasificador incluye **Test-Time Augmentation (TTA)** por defecto: al predecir una
imagen real, evalúa `4` vistas deterministas en memoria (original, espejo horizontal,
recorte central del `86 %` y normalización de contraste `autocontrast`) y promedia sus
probabilidades.

1. **Probar una fotografía individual (celular, cámara o archivo suelto):**
   ```bash
   frutas-data predict-classic ruta/a/imagen.jpg
   ```
   Si se desea desactivar TTA para comparar una sola vista:
   ```bash
   frutas-data predict-classic --no-tta ruta/a/imagen.jpg
   ```

2. **Probar una carpeta completa de fotografías por lotes:**
   ```bash
   frutas-data predict-classic ruta/a/carpeta_fotos/ --output artifacts/mis_predicciones.csv
   ```
   - Si las imágenes están organizadas dentro de subcarpetas `fruta/` y `verdura/` (o en
     carpetas con nombres de categorías registradas en `configs/label_mapping.csv`, como
     `Carrots`, `Garlic`, `Tomato`, `Apple`), el comando detecta automáticamente la clase
     esperada y reporta `accuracy`, `balanced_accuracy`, `f1_macro`,
     `accuracy_when_confident` y la matriz de confusión del lote.

3. **Localizar categorías originales con menor exactitud en `test` o `external`:**
   ```bash
   frutas-data evaluate-classic \
     --split test \
     --output artifacts/error_analysis_by_label.csv
   ```

4. **Ejecutar la prueba de estrés ante 8 condiciones explícitas de ruido:**
   ```bash
   frutas-data evaluate-noise \
     --split test \
     --max-per-source-label 15 \
     --output artifacts/noise_robustness.csv
   ```
   Evalúa `clean`, `gaussian_noise`, `gaussian_blur`, `illumination_low`,
   `illumination_high`, `rotation_crop`, `occlusion_patch` y `combined_heavy`,
   calculando el porcentaje de retención de `F1 macro` respecto a la imagen limpia.

Si la probabilidad mayor no alcanza `0.60`, el resultado incluye
`"uncertain": true`. Esto permite revisar casos difíciles sin inventar una tercera
clase, aunque la salida principal siga siendo fruta o verdura.

## 13.2 Interfaz web y recolección supervisada de ejemplos

La interfaz se instala junto con las dependencias opcionales de la aplicación:

```bash
python -m pip install -e ".[app]"
frutas-app
```

El navegador permite:

1. Subir una imagen `JPG`, `PNG`, `WEBP`, `BMP` o `TIFF` de hasta `20 MB`.
2. Activar o desactivar TTA.
3. Ver la clase predicha, confianza, probabilidades y estado de incertidumbre.
4. Confirmar manualmente la etiqueta correcta y guardar la imagen para un
   reentrenamiento posterior.

El cuarto punto es deliberadamente supervisado. La predicción del modelo **nunca se
usa como etiqueta automática**. Al confirmar un ejemplo, se normaliza a RGB, se
deduplica mediante SHA-256 y se guarda en:

```text
data/raw/team_photos_train/fruta/ui_<hash>.jpg
data/raw/team_photos_train/verdura/ui_<hash>.jpg
```

Guardar una imagen no cambia el modelo cargado. Cuando exista un lote suficientemente
revisado, se incorpora con:

```bash
frutas-data all
frutas-data train-classic
```

`team_photos_train` declara `fixed_split: train`; por lo tanto, ejemplos escogidos a
partir de errores observados en la interfaz no contaminan validation ni test. Las fotos
de evaluación final continúan exclusivamente en `data/external/team_photos/` y nunca
se guardan mediante este formulario.

## 14. Métricas obligatorias

No se reporta solamente accuracy. El resultado final debe incluir:

- Accuracy.
- Balanced accuracy.
- Precision por clase y macro.
- Recall por clase y macro.
- F1 por clase y macro.
- Matriz de confusión.
- Retención de F1 macro bajo las 8 condiciones de ruido (`artifacts/noise_robustness.csv`).
- Resultados separados para test interno y test externo.

## 15. Cuándo está lista la base de datos

La etapa de datos se considera aprobada cuando:

1. El criterio botánico está documentado y aplicado a todas las fuentes.
2. No quedan etiquetas contradictorias dentro del entrenamiento.
3. No existen fugas de duplicados.
4. Train, validation y test contienen ambas clases.
5. Ninguna fuente domina por completo una sola clase.
6. Existe un test externo congelado.
7. El reporte de auditoría puede regenerarse desde cero.
8. Todos los integrantes obtienen los mismos splits usando la misma semilla.

## 16. Registro de avances

### Versión 0.2 — Simplificación binaria

- Alcance reducido a fruta/verdura.
- Sustituida la taxonomía jerárquica por `classes.csv`.
- Manifiesto reducido a una sola `class_id`.
- Duplicados y splits adaptados a las dos clases.
- Baseline convertido a clasificación binaria.
- Adoptado y documentado un único criterio botánico.
- Separadas fotografías del equipo para entrenamiento y prueba externa.

### Versión 0.3 — Entrenamiento clásico completo

- Eliminada la planificación de redes neuronales.
- Implementada extracción numérica reproducible con HSV, HOG y LBP.
- Implementado aumento de ruido en memoria solo para entrenamiento.
- Agregada comparación de regresión logística, SGD, SVM y Extra Trees.
- Agregado ensamble por votación suave de los tres mejores candidatos.
- Selección aislada en validation y evaluación final en test.
- Guardado reproducible del modelo, métricas y tabla de selección.

### Versión 0.4 — Preparación multifuente

- Drive reducido exclusivamente a imágenes y datasets.
- Agregado desglose reproducible de errores por fuente y etiqueta original.
- Los splits existentes se preservan al agregar una fuente nueva.
- Incorporado límite por fuente/etiqueta para evitar dominancia de datasets masivos.
- Fijada la versión 5 y completado el mapeo de `fruits_youssef`.
- Agregadas pruebas de regresión para preservación de splits y evaluación detallada.

### Versión 0.5 — Integración de escenas reales LVIS (YOLO)

- Descarga reproducible desde Hugging Face con commit fijado y metadatos Git sin LFS.
- Conversión automática de cajas YOLO a 91,378 recortes con 15% de contexto (`prepare-lvis`).
- Optimización del detector de duplicados perceptuales mediante segmentación de bits.
- Preservación de `original_split` en datasets publicados.

### Versión 0.6 — Escalado balanceado por bloques y sincronización multifuente

- Enriquecimiento del vector de características a **2,068 dimensiones** incorporando histogramas y momentos espaciales **HSV + CIE Lab** (globales y centrales) y **LBP** (global y central).
- Implementación de `BlockBalancedScaler` para equilibrar la varianza total entre los bloques de color, forma (HOG) y textura (LBP), evitando que HOG domine el 96% de la distancia.
- Migración de `SVC(probability=True)` a `CalibratedClassifierCV(SVC(...), cv=3, ensemble=False)` y regularización de `ExtraTreesClassifier` (`min_samples_leaf=5`), eliminando advertencias de deprecación en `scikit-learn 1.9+`.

### Versión 1.0.0 — Versión entregable robusta, datasets adicionales, TTA y benchmark de ruido

- **Nuevos datasets integrados:**
  - `grocery_store` (`marcusklasson/GroceryStoreDataset`): `5,421` fotografías reales tomadas con cámara de teléfono en supermercados (81 clases finas mapeadas bajo criterio botánico; 49 clases aprobadas con `3,593` fotos y 32 clases de envases/hongos rechazadas).
  - `vegetable_ahmed` (`misrakahmed/vegetable-image-dataset`): `21,000` imágenes en 15 categorías registradas y mapeadas al 100 % en `configs/label_mapping.csv`.
- **Soporte de descarga Git reproducible** (`git_repo` y `git_include_paths` en `src/frutas/download.py`).
- **Muestreo balanceado `50/50` para `ExtraTreesClassifier`** en `_fit_subset`, elevando el recall de `verdura` en `test` de `58.95 %` (`v0.5.0`) / `65.32 %` (`v0.6.0`) a **`74.95 %`** (`v1.0.0`).
- **Inferencia con Test-Time Augmentation (TTA)** de 4 vistas y soporte de **predicción por lotes sobre directorios completos** en `frutas-data predict-classic` con evaluación automática de carpetas etiquetadas.
- **Nuevo comando `frutas-data evaluate-noise`** que certifica una **retención media de F1 macro del `98.00 %`** frente a 7 condiciones de ruido adverso (`artifacts/noise_robustness.csv`).

### Versión 1.1.0 — Quinta fuente activa y calibración de la decisión

- Descargado y auditado `vegetable_ahmed` versión 1: `21,000` imágenes, licencia
  CC BY-SA 4.0 y mapeo botánico completo de sus 15 categorías.
- Inventario total: `193,708` registros, `193,706` imágenes válidas y cero grupos
  duplicados cruzando particiones.
- Agregado un umbral de decisión binario seleccionado únicamente con `validation`;
  el modelo ganador eligió `0.49` para la probabilidad de `fruta`.
- Resultado multifuente en `test` muestreado (`7,881` imágenes): accuracy `0.8126`,
  balanced accuracy `0.8007`, F1 macro `0.7768`, recall `fruta` `0.8261` y recall
  `verdura` `0.7752`.
- Ablación sobre las `13,315` imágenes de test pertenecientes a las cuatro fuentes
  anteriores: accuracy `0.8034` y F1 macro `0.7258`. La balanced accuracy fue `0.7455`,
  por lo que no se debe ocultar el sesgo residual hacia `fruta` y la evaluación externa
  balanceada sigue siendo requisito de cierre.
- Añadida `--exclude-source` a `evaluate-classic` para repetir comparaciones justas sin
  que las imágenes de la fuente nueva inflen el resultado.

### Versión 2.0.0 — Salto a ~90 % de accuracy, descriptor `v4` (1,405 vars), Gradient Boosting y limpieza LVIS

- **Diagnóstico y solución del techo de `~80 %`:**
  1. **Limpieza automática de recortes contaminados en `lvis_fruits_vegetables`:** el `59.2 %` de los recortes LVIS eran menores a `64x64` px y el `25.5 %` (`22,182` recortes) provenían de escenas mixtas (`fruta + verdura` simultáneas en ensaladas o platos) con traslape de cajas. `_eligible` ahora filtra recortes LVIS exigiendo resolución mínima `64x64`, escenas de clase pura (`scene_classes == 1`) y máximo `8` objetos por escena (`scene_boxes <= 8`).
  2. **Reingeniería del descriptor clásico a `1,405` características (`v4`):** se eliminó la inanición de atributos de color/textura en árboles (donde el `85.3 %` de las columnas eran celdas HOG crudas) mediante un descriptor balanceado con `839` variables de color/cromaticidad/índices botánicos/separación figura-fondo (`59.7 %`), `458` de bordes Sobel/Laplaciano y HOG multiescala agrupado (`32.6 %`) y `108` de textura LBP en 3 escalas (`7.7 %`).
  3. **Incorporación de `HistGradientBoostingClassifier` (`hgb_depth_63`, `hgb_depth_95`) y `AdaptivePCA(256) + SVC(RBF)` (`14,000` filas)** dentro de un ensamble ponderado de los 4 mejores modelos (`soft_voting_top3`).
  4. **Sexta fuente configurada (`fruits360`):** soporte de `git sparse-checkout` en `src/frutas/download.py` y mapeo de 44 categorías estrictamente balanceadas (`20` verduras botánicas y `24` frutas botánicas) en `configs/datasets.yaml` y `configs/label_mapping.csv`.
- **Salto verificado en métricas (`test` multifuente `9,293` imgs / `test` elegible completo `13,141` imgs):**
  - Accuracy en `test` muestreado: **`89.65 %`** (`+8.39 pts` vs `81.26 %`), Balanced Accuracy: **`87.06 %`** (`+6.99 pts`), F1 Macro: **`0.8701`** (`+9.33 pts`), Recall `fruta` / `verdura`: **`92.78 %`** / **`81.34 %`**.
  - Accuracy en `test` completo con TTA (`13,141` imgs): **`89.13 %`** (`vegetable_ahmed`: **`97.69 %`**, `grocery_store`: **`91.17 %`**, `fruits_youssef`: **`86.97 %`**, `kritikseth36`: **`79.55 %`**, `lvis_fruits_vegetables`: **`77.03 %`**).
  - Ablación sin `vegetable_ahmed` (`10,976` imgs): Accuracy **`86.63 %`**, Balanced Accuracy **`80.34 %`**, F1 Macro **`0.8045`**.

### Versión 2.1.0 — Interfaz web y feedback supervisado

- Agregada la aplicación local `frutas-app` con carga y vista previa de imágenes,
  predicción TTA, probabilidades por clase e indicador de incertidumbre.
- Agregado guardado opcional de ejemplos revisados en `team_photos_train`, con
  validación de formato, límite de `20 MB`, conversión RGB y deduplicación SHA-256.
- La fuente de feedback queda fijada exclusivamente a `train`, evitando contaminación
  de validation y test.
- Agregadas pruebas de decodificación, clasificación, deduplicación, rechazo de entradas
  inválidas y asignación forzada a train.

### Próximo hito operativo del equipo

Reunir las fotografías propias tomadas con celular en `data/external/team_photos/fruta/`
y `data/external/team_photos/verdura/` y ejecutar `frutas-data predict-classic data/external/team_photos`
o `frutas-data evaluate-classic --split external` para documentar la generalización en campo.

## 17. Ejecución verificada inicial (solo `kritikseth36`) — 15 de octubre de 2026

En la etapa `v0.3`, la versión 8 de `kritikseth36` (3,825 imágenes) obtuvo en test
interno (`577` imágenes): Accuracy `0.7175`, Balanced accuracy `0.7105`, F1 macro
`0.7064` y Accuracy en predicciones confiables `0.7655` (respaldado en
`artifacts/classical_model_v0.3_seed42.zip`).

## 18. Segundo dataset: `fruits_youssef`

La fuente `fruits_youssef` (versión 5, 72,084 imágenes en 50 categorías, licencia CC0)
aporta 34 etiquetas mapeadas a fruta y 14 a verdura (`Mushroom` y `Peanuts` quedan
rechazadas por estar fuera del alcance botánico binario).

## 19. Tercer dataset: `LVIS_Fruits_And_Vegetables` con escenas reales y ruido

La fuente `henningheyen/LVIS_Fruits_And_Vegetables` (commit
`51362ad3532292d29e63f9e46c2cd33945ffe6b5`, licencia Apache-2.0) aporta 8,401 escenas
con cajas YOLO. El comando `prepare-lvis` genera **91,378 recortes** con 15% de margen
de contexto en `data/derived/lvis_fruits_vegetables/` y registra su trazabilidad en
`crops_manifest.csv`. Se rechazan `almond`, `bean curd/tofu`, `chickpea`, `mushroom` y
`pickle`.

## 20. Cuarto, quinto y sexto datasets: `grocery_store`, `vegetable_ahmed` y `fruits360`

- **`grocery_store` (`marcusklasson/GroceryStoreDataset`):** `5,421` imágenes tomadas con smartphones en supermercados reales a través de 81 clases finas (`Fruit`, `Vegetables`, `Packages`). Las 31 clases de `Packages` (leches, jugos, yogures) y `Brown-Cap-Mushroom` se excluyen (`rejected`), dejando **3,593 fotografías reales aprobadas** en `train` (`1,737`), `validation` (`191`) y `test` (`1,665`).
- **`vegetable_ahmed` (`misrakahmed/vegetable-image-dataset`, versión 1):** `21,000` imágenes en 15 clases (`Broccoli`, `Cabbage`, `Carrot`, `Cauliflower`, `Potato`, `Radish` como `verdura`; `Bean`, `Bitter_Gourd`, `Bottle_Gourd`, `Brinjal`, `Capsicum`, `Cucumber`, `Papaya`, `Pumpkin`, `Tomato` como `fruta` botánica). Está descargado, auditado y se reproduce con `frutas-data download vegetable_ahmed`.
- **`fruits360` (`fruits-360/fruits-360-100x100`, commit `c23fa253bb3281960a741f51be86824c17bb2244`):** configurado mediante `git sparse-checkout` para descargar únicamente un subconjunto curado y balanceado de **44 categorías** (`20` de `verdura` botánica y `24` de `fruta` botánica) con `frutas-data download fruits360`.

## 21. Ejecución histórica (`v1.0.0` — 4 datasets activos en disco)

Con las cuatro fuentes habilitadas (`lvis_fruits_vegetables`, `fruits_youssef`,
`grocery_store` y `kritikseth36`), el inventario y auditoría en
`data/interim/data_report.md` registran:

- **172,708 registros totales** (`172,706` imágenes válidas y `2` corruptas excluidas).
- **165,024 imágenes aprobadas** (`7,684` rechazadas por categoría fuera de alcance y `143` excluidas por conflicto de etiqueta entre duplicados).
- **126,447 imágenes en `train`, 25,118 en `validation` y 13,316 en `test`.**
- **0 grupos duplicados con fuga entre particiones.**

Al ejecutar `frutas-data train-classic --image-size 64 --augmentations 1 --max-per-source-label 100 --svm-max-rows 6000 --seed 42`, se muestrearon `14,813` imágenes de `train` (`29,626` con aumento de ruido), `7,279` de `validation` y `6,383` de `test`. El modelo ganador seleccionado en validación fue `extra_trees_040` (`F1 macro = 0.6965`, `Balanced Accuracy = 0.7181` en validación).

Resultados verificados en `test` muestreado (`6,383` imágenes) y en `test` completo (`13,316` imágenes):

| Métrica | `test` muestreado (`6,383` imgs) | `test` completo (`13,316` imgs) |
|---|---:|---:|
| Accuracy | **0.7725** (`77.25 %`) | **0.7854** (`78.54 %`) |
| Balanced accuracy | **0.7645** (`76.45 %`) | **0.7610** (`76.10 %`) |
| Precision macro | **0.7103** (`71.03 %`) | **0.7038** (`70.38 %`) |
| Recall macro | **0.7645** (`76.45 %`) | **0.7610** (`76.10 %`) |
| F1 macro | **0.7235** (`72.35 %`) | **0.7201** (`72.01 %`) |
| Recall `fruta` / `verdura` | **77.96 %** / **74.95 %** | **80.31 %** / **71.89 %** |
| Accuracy en predicciones confiables (`>= 0.60`) | **0.8772** (`87.72 %`) | **0.8958** (`89.58 %`) |

## 22. Ejecución histórica (`v1.1.0` — 5 datasets activos antes del rediseño `v2.0.0`)

En `v1.1.0`, tras incorporar `vegetable_ahmed` pero antes de limpiar los recortes contaminados de `LVIS` y antes del descriptor `v4` (`1,405` vars) + `HistGradientBoosting`, el modelo `extra_trees_040` obtuvo en `test` muestreado (`7,881` imgs): Accuracy `0.8126`, Balanced accuracy `0.8007` y F1 macro `0.7768` (`0.8034` accuracy y `0.7258` F1 macro en las fuentes previas sin `vegetable_ahmed`).

## 23. Ejecución actual verificada (`v2.0.0` — Descriptor `v4`, Limpieza LVIS y Ensamble con Gradient Boosting)

Con el comando por defecto:

```bash
frutas-data train-classic
```

Se entrenan los 8 candidatos clásicos sobre `19,097` imágenes de `train`, se evalúan en `9,422` imágenes de `validation` y se reentrena el ensamble ganador (`soft_voting_top3`, umbral `0.485`) sobre las `28,519` imágenes combinadas (`train + validation`).

### Tabla de selección de modelos en `validation` (`9,422` imágenes, `artifacts/model_selection.csv`)

| Modelo | F1 Macro | Balanced Accuracy | Accuracy | Precision Macro | Recall Macro | Umbral |
|---|---:|---:|---:|---:|---:|---:|
| **`soft_voting_top3` (Ganador)** | **0.8766** | **0.8727** | **0.8980** | **0.8808** | **0.8727** | `0.485` |
| `hgb_depth_95` | 0.8708 | 0.8705 | 0.8922 | 0.8711 | 0.8705 | `0.545` |
| `hgb_depth_63` | 0.8705 | 0.8615 | 0.8946 | 0.8815 | 0.8615 | `0.445` |
| `svm_rbf_c10` (`PCA-256`) | 0.8509 | 0.8505 | 0.8756 | 0.8513 | 0.8505 | `0.425` |
| `svm_rbf_c25` (`PCA-256`) | 0.8494 | 0.8419 | 0.8770 | 0.8582 | 0.8419 | `0.375` |
| `extra_trees_040` | 0.8436 | 0.8318 | 0.8742 | 0.8593 | 0.8318 | `0.540` |
| `extra_trees_sqrt` | 0.8328 | 0.8331 | 0.8602 | 0.8325 | 0.8331 | `0.575` |
| `logistic_c1` | 0.7657 | 0.7766 | 0.7971 | 0.7584 | 0.7766 | `0.420` |
| `sgd_log` | 0.7652 | 0.7779 | 0.7955 | 0.7571 | 0.7779 | `0.425` |

### Resultados en `test` (`9,293` muestreadas, `13,141` completas con TTA y `10,976` sin `vegetable_ahmed`)

| Métrica | `test` muestreado (`9,293` imgs) | `test` completo con TTA (`13,141` imgs) | Ablación sin `vegetable_ahmed` (`10,976` imgs) |
|---|---:|---:|---:|
| **Accuracy** | **0.8965** (`89.65 %`) | **0.8913** (`89.13 %`) | **0.8663** (`86.63 %`) |
| **Balanced accuracy** | **0.8706** (`87.06 %`) | **0.8461** (`84.61 %`) | **0.8034** (`80.34 %`) |
| **Precision macro** | **0.8695** (`86.95 %`) | **0.8593** (`85.93 %`) | **0.8057** (`80.57 %`) |
| **Recall macro** | **0.8706** (`87.06 %`) | **0.8461** (`84.61 %`) | **0.8034** (`80.34 %`) |
| **F1 macro** | **0.8701** (`87.01 %`) | **0.8523** (`85.23 %`) | **0.8045** (`80.45 %`) |
| **Recall `fruta` / `verdura`** | **92.78 %** / **81.34 %** | **93.65 %** / **75.56 %** | **91.58 %** / **69.10 %** |
| **Accuracy confiable (`>= 0.60`)** | **0.9197** (`91.97 %`) | **0.9198** (`91.98 %`) | **0.8948** (`89.48 %`) |

### Benchmark de robustez ante ruido (`artifacts/noise_robustness.csv`, `1,677` imágenes de `test` con TTA)

| Condición | Accuracy | Balanced Accuracy | F1 Macro | Exactitud Confiable (`>=0.60`) | Retención de F1 |
|---|---:|---:|---:|---:|---:|
| `clean` | **87.00 %** | **82.74 %** | **0.8303** | **90.15 %** | **100.00 %** |
| `occlusion_patch` | 86.52 % | 80.51 % | 0.8179 | 89.39 % | 98.50 % |
| `illumination_high` | 85.39 % | 78.42 % | 0.7998 | 88.50 % | 96.32 % |
| `rotation_crop` | 85.33 % | 81.17 % | 0.8108 | 87.98 % | 97.65 % |
| `gaussian_noise` | 84.68 % | 76.39 % | 0.7844 | 88.16 % | 94.47 % |
| `combined_heavy` | 83.60 % | 78.01 % | 0.7845 | 88.21 % | 94.48 % |
| `gaussian_blur` | 82.59 % | 79.24 % | 0.7822 | 85.92 % | 94.20 % |
| `illumination_low` | 81.57 % | 79.51 % | 0.7758 | 86.07 % | 93.43 % |
| **Media bajo ruido** | **84.24 %** | **79.04 %** | **0.7936** | **87.75 %** | **95.58 %** |
