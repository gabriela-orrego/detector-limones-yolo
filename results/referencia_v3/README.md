# Corrida de referencia (v3 del grupo de Robótica, 12/11/2026)

Figuras y tablas extraídas del cuaderno ejecutado
[`v3_tarea_va_limones_amarillos_2026-11-12.ipynb`](https://github.com/HecVelaz/recolector_de_frutas/blob/main/vision/entrenamiento_limon/versiones/v3_tarea_va_limones_amarillos_2026-11-12.ipynb)
del repositorio `HecVelaz/recolector_de_frutas`. Es la misma configuración que el modelo base de este proyecto
(notebook 02): YOLO26n, 640 px, hasta 60 épocas; paró en la época 51 (mejor: 31) tras 14,5 minutos en una GPU T4.

| Archivo | Qué muestra |
|---|---|
| `dataset.csv` | Imágenes y limones de cada conjunto |
| `evaluacion.csv` | Precisión, recobrado y mAP |
| `muestras_dataset.png` | Ejemplos con sus etiquetas (arriba, dataset base; abajo, fotos propias) |
| `curvas_entrenamiento.png` | Pérdidas y métricas por época |
| `matriz_confusion.png` | 110 aciertos, 6 limones no detectados y 14 detecciones de más, sobre 116 limones |
| `curvas_pr_f1.png` | Curvas precisión–recobrado y F1–confianza |
| `conteo_por_foto.png` | Limones reales y detectados en cada una de las 20 fotos propias |
