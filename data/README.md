# Contrato de datos del clasificador binario

- `raw/`: descargas originales e inmutables. Nunca se editan ni renombran.
- `derived/`: transformaciones reproducibles, como los recortes YOLO de LVIS.
- `interim/`: manifiestos, candidatos duplicados y reportes de auditoría.
- `processed/`: índices finales consumidos por entrenamiento; no contiene copias
  de imágenes salvo que una etapa posterior lo requiera explícitamente.
- `external/`: fotografías reservadas para la evaluación final. Nunca se usan
  para entrenar ni para elegir hiperparámetros.
- `quarantine/`: destino opcional para una revisión humana futura. El pipeline
  actual no mueve archivos aquí automáticamente.

Cada dataset debe conservar su estructura original dentro de una carpeta propia.
Ejemplo:

```text
data/raw/fruits262/
data/raw/fruits360/
data/raw/team_photos_train/fruta/
data/raw/team_photos_train/verdura/
data/external/team_photos/fruta/
data/external/team_photos/verdura/
```

Las especies pueden conservarse en `original_label` para auditoría, pero el único
objetivo de aprendizaje es `class_id`, cuyos valores válidos son `fruta` y
`verdura`.

Google Drive se utiliza únicamente como origen compartido de imágenes. Al descargar
las carpetas `team_photos_train` y `team_photos_external`, sus contenidos se colocan
en las rutas locales mostradas arriba. Código, modelos, reportes y manifiestos no se
guardan en Drive.

