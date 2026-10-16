# Guía del equipo: clasificador fruta o verdura

## 1. Alcance oficial

El sistema recibe una imagen y produce exactamente una predicción binaria:

```text
imagen -> fruta | verdura
```


El objetivo técnico es que el clasificador mantenga buen desempeño con ruido real:
fondos diferentes, cambios de iluminación, rotación, recortes, desenfoque,
compresión, objetos parcialmente tapados y frutas o verduras cortadas.

## 2. Criterio oficial de etiquetado

El proyecto adopta el **criterio botánico**, La regla se aplica de forma uniforme a todas las fuentes:

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
- Documentación y, opcionalmente, manifiestos pequeños sin rutas privadas.

No se versionan:

- Imágenes.
- Archivos ZIP de datasets.
- Características numéricas.
- Modelos entrenados.
- Tokens de Kaggle o GitHub.

### Google Drive compartido

Se almacenan los archivos pesados que pertenecen al equipo:

```text
clasificador-frutas-verduras-datos/
├── team_photos_train/
│   ├── fruta/
│   └── verdura/
├── team_photos_external/
│   ├── fruta/
│   └── verdura/
├── manifests/
├── features/
└── models/
```

Los datasets públicos reproducibles se descargan desde su fuente. No es necesario
duplicarlos en Drive, salvo que la fuente deje de estar disponible.

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

- Fruit and Vegetable Image Recognition, con 36 etiquetas originales.
- Fruits-360.
- Fruits-262.
- Fruit and Vegetables Classification.
- Fuente compartida por Google.
- Fotografías del equipo.

No es necesario usar todos los datasets desde el primer entrenamiento. El orden
recomendado es:

1. Comenzar con `kritikseth36` porque contiene ambas clases y escenarios variados.
2. Agregar fotografías propias de entrenamiento con ruido real.
3. Auditar balance y errores.
4. Incorporar otra fuente solamente si cubre una carencia concreta.

Una fuente que contiene exclusivamente frutas, como puede ocurrir con algunos
datasets, no debe añadirse sin verduras comparables. De lo contrario, el modelo
puede aprender el estilo de la fuente en vez de la diferencia fruta/verdura.

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
data/raw/google_shared/
```

El dataset inicial se descarga con un solo comando:

```bash
frutas-data download kritikseth36
```

La configuración fija la versión 8 del dataset. Por eso todos descargan exactamente
la misma versión y no dependen de cuál sea la versión más reciente en el futuro. Una
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
frutas-data download kritikseth36
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
- SHA-256 y hash perceptual.
- Split original, cuando existe.

Una imagen corrupta se registra, pero no detiene todo el inventario.

### 9.2 Duplicados

`dedupe` no borra archivos. Agrupa:

- Copias exactas mediante SHA-256.
- Copias visualmente parecidas mediante hash perceptual.

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

Las particiones se estratifican por `class_id`, no por especie.

### 9.4 Auditoría

`audit` genera `data/interim/data_report.md`. Antes de entrenar se debe verificar:

- Cero fugas de grupos duplicados entre splits.
- Cero conflictos de etiqueta pendientes.
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
- Rotación moderada.
- Cambio de brillo, contraste y saturación.
- Desenfoque gaussiano o de movimiento.
- Ruido de sensor.
- Oclusión parcial.

Estas variantes se generan en memoria y no duplican archivos dentro de `data/raw/`.

Los aumentos no sustituyen fotografías reales con fondos variados.

## 12. Baseline clásico

El baseline convierte las imágenes en un vector numérico:

- HSV describe distribución de color.
- HOG describe bordes y formas.
- LBP describe textura local.

Después entrena un clasificador lineal con normalización y pesos balanceados:

```bash
frutas-data train-baseline --image-size 64 --max-per-class 1000
frutas-data predict-baseline ruta/a/imagen.jpg
```

La predicción contiene:

```json
{
  "prediction": "fruta",
  "confidence": 0.83,
  "probabilities": {"fruta": 0.83, "verdura": 0.17},
  "uncertain": false
}
```

`uncertain` es una advertencia de baja confianza; la predicción sigue siendo binaria.

## 13. Selección y ensamble clásico

El entrenamiento final se ejecuta con:

```bash
frutas-data train-classic --image-size 64 --augmentations 1 --seed 42
```

El flujo evita escoger un modelo mirando los resultados de prueba:

```text
train -> extracción HSV + HOG + LBP -> candidatos clásicos
validation -> selección por F1 macro
train + validation -> reentrenamiento del ganador
test -> evaluación interna una sola vez
external -> evaluación separada, si existen fotografías externas
```

Los candidatos son regresión logística, SGD con pérdida logística, SVM RBF y Extra
Trees. También se evalúa un ensamble de votación suave con los tres candidatos
individuales mejor posicionados. Todos manejan el desbalance de clases y exponen
probabilidades comparables.

La selección prioriza F1 macro, con balanced accuracy y accuracy como desempates. El
comando crea:

- `artifacts/classical_model.joblib`: modelo y configuración de características.
- `artifacts/classical_metrics.json`: métricas completas y matriz de confusión.
- `artifacts/model_selection.csv`: tabla ordenada de candidatos.

Para probar una imagen nueva:

```bash
frutas-data predict-classic ruta/a/imagen.jpg
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

También se recomienda evaluar por condición de ruido: limpia, cortada, fondo complejo,
desenfoque, iluminación difícil y oclusión.

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

### Próximo hito

Agregar y congelar fotografías externas del equipo, volver a generar el manifiesto y
medir generalización fuera del dataset público sin cambiar hiperparámetros mirando
esas fotografías.

## 17. Ejecución verificada — 15 de octubre de 2026

La versión 8 de `kritikseth36` se descargó y procesó de extremo a extremo con semilla
42. El inventario real obtuvo:

- 3,825 imágenes válidas de 3,825.
- 36 etiquetas originales aprobadas.
- 2,685 imágenes en train, 563 en validation y 577 en test.
- 1,741 frutas y 944 verduras en train.
- 360 grupos candidatos a duplicado.
- Cero conflictos de etiqueta y cero fugas entre particiones.
- Las 36 etiquetas originales están representadas en los tres splits.

El entrenamiento utilizó una variante de ruido por cada imagen de train. El ensamble
de votación suave fue seleccionado solamente con validation. El resultado congelado
en test interno fue:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7175 |
| Balanced accuracy | 0.7105 |
| Precision macro | 0.7044 |
| Recall macro | 0.7105 |
| F1 macro | 0.7064 |
| Cobertura con confianza >= 0.60 | 0.8648 |
| Accuracy entre predicciones confiables | 0.7655 |

La matriz de confusión, con filas reales y columnas predichas en el orden
`[fruta, verdura]`, fue:

```text
[[263,  92],
 [ 71, 151]]
```

Estos números son la línea base experimental vigente, no una promesa para fotos
nuevas. Aún no existe un resultado externo porque la carpeta
`data/external/team_photos/` está vacía. Esa evaluación es el siguiente control
obligatorio antes de presentar el modelo como robusto.
