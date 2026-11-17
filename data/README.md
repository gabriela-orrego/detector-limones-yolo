# Datos

Las imágenes no se guardan en el repositorio: se descargan de Roboflow Universe con el notebook
`01_preprocesamiento.ipynb`, que deja acá `datos_limon.zip` (el dataset limpio y unificado).

| Dataset | Fuente | Contenido |
|---|---|---|
| Base | [brinjalproject/lemons-segmentation](https://universe.roboflow.com/brinjalproject/lemons-segmentation), versión 5 | Limones en el árbol, etiquetados con polígonos en 3 clases. Se usa solo la clase de limones amarillos: 915 imágenes, 1 315 limones |
| Propio | [hectors-workspace-e5dhh/limones-amarillos](https://universe.roboflow.com/hectors-workspace-e5dhh/limones-amarillos), versión 3 (CC BY 4.0) | 50 fotos de un limonero en maceta, 244 limones etiquetados con cajas: 30 train / 10 valid / 10 test |

Estructura de `datos_limon.zip`:

```
base/{train,valid,test}/{images,labels}      dataset público, una sola clase (0 = limon)
propio/{train,valid,test}/{images,labels}    fotos propias
```
