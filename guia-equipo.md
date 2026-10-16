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
├── tests/
└── artifacts/               # modelos y métricas locales
```

No se renombran, convierten o sobrescriben originales dentro de `data/raw/`.

## 5. Fuentes de datos

Las fuentes conocidas están registradas en `configs/datasets.yaml`:

- `kritikseth36`: Fruit and Vegetable Image Recognition (36 etiquetas originales, activo).
- `fruits_youssef`: Fruit and Vegetables Classification (50 categorías, versión 5, activo).
- `lvis_fruits_vegetables`: LVIS Fruits And Vegetables (61 etiquetas canónicas de detección YOLO convertidas a recortes en `data/derived/`, activo).
- `fruits360`: Fruits-360 (inactivo por fondo blanco controlado).
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
python -m pip install -e ".[dev,notebooks]"
pytest
```

En Windows, la activación normalmente es:

```powershell
.venv\Scripts\Activate.ps1
```

La instalación se realiza desde `pyproject.toml`; no se deben instalar versiones al
azar en cada computadora ni mantener listas distintas por integrante.

## 7. Colocación de datasets

Cada dataset conserva su estructura original en la ruta indicada por su entrada de
`configs/datasets.yaml`:

```text
data/raw/kritikseth36/
data/raw/fruits360/
data/raw/fruits262/
data/raw/fruits_youssef/
data/raw/lvis_fruits_vegetables/
data/derived/lvis_fruits_vegetables/
data/raw/google_shared/
```

Los tres datasets públicos activos se descargan y preparan con:

```bash
frutas-data download kritikseth36
frutas-data download fruits_youssef
frutas-data download lvis_fruits_vegetables
frutas-data prepare-lvis
```

La configuración fija la versión o commit de cada dataset. Por eso todos descargan
exactamente los mismos archivos y no dependen de cambios futuros en las fuentes. Una
descarga completa se reutiliza; `--force` solamente se emplea cuando se verificó que
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

## 12. Extracción de características y baseline clásico

Cada imagen se redimensiona a `64x64` y se transforma en un vector numérico
determinista de **2,068 características** (`float32`):

- **Bloque de Color (252 dimensiones):** histogramas de 16 bins en los espacios **HSV** y **CIE Lab** tanto de la imagen completa como del recorte central (`50%` interior, para aislar el objeto del fondo), más momentos espaciales (media y desviación estándar) en una cuadrícula `2x2` y región central.
- **Bloque de Forma (1,764 dimensiones):** descriptor **HOG** sobre imagen en escala de grises ecualizada.
- **Bloque de Textura (52 dimensiones):** histogramas de **LBP uniforme** (`P=24, R=3`) global y central.

Los clasificadores lineales y SVM utilizan `BlockBalancedScaler`, que estandariza cada
columna y pondera cada bloque por el inverso de la raíz cuadrada de su dimensión
(`1 / sqrt(d_bloque)`), garantizando que color, forma y textura aporten exactamente la
misma varianza total (`1.0` por bloque).

Para ejecutar el baseline lineal rápido:

```bash
frutas-data train-baseline --image-size 64 --max-per-class 1000
frutas-data predict-baseline ruta/a/imagen.jpg
```

## 13. Selección y ensamble clásico

El entrenamiento final se ejecuta con:

```bash
frutas-data train-classic \
  --image-size 64 \
  --augmentations 1 \
  --max-per-source-label 100 \
  --svm-max-rows 6000 \
  --seed 42
```

El flujo evita escoger un modelo mirando los resultados de prueba:

```text
train -> extracción HSV/Lab + HOG + LBP (2,068 vars) -> candidatos clásicos
validation -> selección por F1 macro
train + validation -> reentrenamiento del ganador
test -> evaluación interna una sola vez
external -> evaluación separada, si existen fotografías externas
```

Los candidatos son regresión logística (`C=1`, `C=10`), SGD con pérdida logística,
SVM RBF calibrado (`CalibratedClassifierCV` con `C=10`, `C=30` y `ensemble=False`) y
Extra Trees regularizados (`min_samples_leaf=5`, con `max_features="sqrt"` y `0.25`).
También se evalúa un ensamble de votación suave (`soft_voting_top3`) con los tres
candidatos individuales mejor posicionados.

Los modelos lineales y Extra Trees aprovechan todas las filas seleccionadas. SVM
RBF tiene costo aproximadamente cuadrático, por lo que recibe como máximo 6,000
filas balanceadas y reproducibles.

La selección prioriza F1 macro, con balanced accuracy y accuracy como desempates. El
comando crea:

- `artifacts/classical_model.joblib`: modelo y configuración de características.
- `artifacts/classical_metrics.json`: métricas completas y matriz de confusión.
- `artifacts/model_selection.csv`: tabla ordenada de candidatos.

Para probar una imagen nueva:

```bash
frutas-data predict-classic ruta/a/imagen.jpg
```

Para localizar clases originales con bajo desempeño:

```bash
frutas-data evaluate-classic \
  --split test \
  --output artifacts/error_analysis_by_label.csv
```

Si la probabilidad mayor no alcanza `0.60`, el resultado incluye
`"uncertain": true`. Esto permite revisar casos difíciles sin inventar una tercera
clase, aunque la salida principal siga siendo fruta o verdura.

## 14. Métricas obligatorias

No se reporta solamente accuracy. El resultado final debe incluir:

- Accuracy.
- Balanced accuracy.
- Precision por clase y macro.
- Recall por clase y macro.
- F1 por clase y macro.
- Matriz de confusión.
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
- Reentrenamiento y evaluación completa sincronizados con el `dataset.csv` multifuente (`167,287` registros), elevando el **Accuracy en test de `63.40%` a `76.37%`** y el **F1 macro de `0.6057` a `0.7127`** (`81.75%` de exactitud en predicciones confiables).

### Próximo hito

Agregar y congelar fotografías externas del equipo (`data/external/team_photos/`), volver
a generar el manifiesto y medir generalización fuera de los datasets públicos sin cambiar
hiperparámetros mirando esas fotografías.

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

## 20. Ejecución multifuente verificada (`v0.6.0` — 3 datasets activos)

Con las tres fuentes habilitadas (`kritikseth36`, `fruits_youssef` y
`lvis_fruits_vegetables`), el inventario y auditoría en `data/interim/data_report.md`
registran:

- **167,287 registros totales** (`167,285` imágenes válidas y `2` corruptas excluidas).
- **161,431 imágenes aprobadas** (`5,856` rechazadas por categoría fuera de alcance y `141` excluidas por conflicto de etiqueta entre duplicados).
- **124,710 imágenes en `train`, 24,927 en `validation` y 11,651 en `test`.**
- **0 grupos duplicados con fuga entre particiones.**

Al ejecutar `frutas-data train-classic --image-size 64 --augmentations 1 --max-per-source-label 100 --svm-max-rows 6000 --seed 42`, se muestrearon `13,076` imágenes de `train` (`26,152` con aumento de ruido), `7,088` de `validation` y `4,718` de `test`. El modelo ganador seleccionado en validación fue `soft_voting_top3` (`svm_rbf_c10` + `svm_rbf_c30` + `extra_trees_040`, con `F1 macro = 0.7169` en validación).

El resultado verificado en el conjunto `test` muestreado (`4,718` imágenes) fue:

| Métrica | Resultado (`v0.6.0`) |
|---|---:|
| Accuracy | **0.7637** (`76.37%`) |
| Balanced accuracy | **0.7281** (`72.81%`) |
| Precision macro | **0.7039** (`70.39%`) |
| Recall macro | **0.7281** (`72.81%`) |
| F1 macro | **0.7127** (`71.27%`) |
| Cobertura con confianza >= 0.60 | **0.7641** (`76.41%`) |
| Accuracy entre predicciones confiables | **0.8175** (`81.75%`) |

Matriz de confusión en `test` con filas reales y columnas predichas en el orden
`[fruta, verdura]`:

```text
[[2795,  686],
 [ 429,  808]]
```
