# Fish-classification
Este repositorio consiste en la realización de mi tesis de pregrado para la carrera de Ciencias de la Computación en la Universidad de Tecnología e Ingeniería UTEC. La tesis se titula "Detección y clasificación de especies de peces: Un pipeline eficiente en un dataset no etiquetado". 

La tesis consiste de 3 repositorios diferentes:
- [Fish-Classification](https://github.com/cesar214567/Fish-classification) (Repositorio principal)
- [yolov5](https://github.com/cesar214567/yolov5) (Repositorio fork de ultralytics modificado para el entrenamiento del modelo y exporte del modelo preentrenado.)
- [UniDet](https://github.com/cesar214567/UniDet) (Repositorio fork de xingyizhou/UniDet modificado para el uso del modelo preentrenado)

<details open>
<summary>Instalacion</summary>

```
git clone https://github.com/cesar214567/yolov5
```
\* Si desea replicar el entrenamiento del yolov5 con el dataset usado para la tesis se puede dirigir al siguiente repositorio y dirigirse al readme en la sección entrenamiento: 
```
https://github.com/cesar214567/Fish-classification
```
\* Los modelos preentrenados se encuentran en la carpeta ./models. En el presente proyecto se trabajaron con 3 diferentes arquitecturas, yolov3, yolov5 y UniDet. Debido a que yolov3 consiguió resultados no favorables y también debido al peso de sus modelos entrenados (se probó con 2 modelos pre-entrenados con diferentes versiones de OpenImages), estos fueron excluidos, pero si desea utilizarlos, puede descargarlos desde el siguiente link y incluirlos en esa carpeta:
```
https://drive.google.com/drive/folders/16WkuZiWYL7-6sgTcZkarooj9_yys_Qd4?usp=sharing 
```
</details>

<details open>
<summary>Introducción</summary>


A continuación se explicarán el contenido de cada una de las carpetas en el presente proyecto:

<details open>
<summary>Fish_Dataset</summary>

Contiene imágenes dentro del dataset A Large Scale Fish Dataset, obtenido del siguiente link:
```
https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
```
De este dataset se obtuvieron las 1000 imágenes de la especie trucha marrón, una especie que también coexiste dentro del Perú.
</details>


<details open>
<summary>FishSpecies</summary>

Contiene imágenes dentro del dataset Fish Species, obtenido del siguiente link:
```
https://www.kaggle.com/datasets/giannisgeorgiou/fish-species
```
De este dataset se obtuvieron las 2000 imágenes de las siguientes especies peruanas:

* Mugil cephalus: Lisa
* Rhinobatos cemiculus: Pez guitarra
* Scomber japonicus: Caballa
* Tetrapturus Belone: Pez espada

Para cada uno de ellos, se tuvieron 2 carpetas, uno para entrenamiento y otro para testeo,conteniendo una distribución de 85-15\%, incluyendo tambien las imágenes de Fish_Dataset ya repartidas dentro de estas carpetas.
</details>

<details open>
<summary>NatureConservancy</summary>

Contiene imágenes dentro del dataset The Nature Conservancy Fisheries Monitoring, obtenido del siguiente link:
```
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring
```
De este dataset se obtuvieron un numero diferente de imágenes de las siguientes especies:

* Albacore Tuna (ALB): Bonito.
* Bigeye Tuna (BET): Atun de ojo grande.
* Dolphinfish (DOL): Mero perico o perico.
* Moonfish (LAG): Pez luna (especie no peruana)
* Shark (SHARK): Tiburon blanco, Toyo y Toyo de leche (multiples especies)
* Yellowfin Tuna (YFT): Atun de aleta amarilla
* Other (OTHER): Incluye varias especies excepto las anteriores mencionadas.

Para cada uno de ellos, se tuvieron 2 carpetas, uno para entrenamiento y otro para testeo. 
<br>
Dentro de la carpeta de entrenamiento se tuvieron los siguientes archivos: 
```
./NatureConservancy/train/<clase>/*.jpg 
./NatureConservancy/train/<clase>/*.txt 
```
\* "clase" hace referencia alas iniciales de las clases descritas en la lista anterior.
Para cada imagen se tiene un archivo .txt que hace referencia a los bounding boxes que fueron creados por un experto para cada uno de los peces. En esta carpeta los peces dentro de la imagen ya venian etiquetados con su especie, por lo que únicamente se necesitaron los bounding boxes de parte del experto

De la misma manera, en la carpeta de test_stg1 se encuentra la misma distribución de archivos, pero el dataset no incluía las etiquetas, por lo que todos los archivos no se encuentran separados por sus clases. 

</details>

<details open>
<summary>NatureConservancyCropped</summary>

Contiene las imágenes de la carpeta anterior recortadas y etiquetadas segun la evaluación del experto. De la misma manera, estan separadas en entrenamiento y prueba siguiendo la misma nomenclatura anteriormente mencionada

</details>

<details open>
<summary>NatureConservancyCroppedMerged</summary>

Contiene las imágenes de la carpeta anterior unidas con el dataset FishSpecies para agregar clases al clasificador

</details>

<details open>
<summary>NatureConservancyCroppedAugmented</summary>

Contiene las imagenes anteriormente recortadas, las cuales fueron modificadas usando DataAugmentation para tratar el problema de overfitting generado en cierta parte de la experimentación, pero luego fue dejada apartada ya que al unir las bases de datos de training y testing por motivos de mala distribución de data, se solventó el problema. Queda como evidencia del trabajo realizado.

</details>

<details open>
<summary>models</summary>

Dentro de esta carpeta se contiene los archivos que especifican cada uno de los detectores pre-entrenados:
- yolov3
  - OpenImages V3
    - <pre>yolo.cfg      #Detalla la arquitectura del modelo</pre>
    - <pre>yolo.data     #Detalla la informacion con la que se entreno</pre>
    - <pre>yolo.names    #Detalla las clases detectables</pre>
  - OpenImages V6
    - <pre>yolo2.cfg     #Detalla la arquitectura del modelo</pre>
    - <pre>yolo2.data    #Detalla la informacion con la que se entreno</pre>
    - <pre>yolo2.names   #Detalla las clases detectables</pre>
- yolov5
  - Objects365 pretrained
    - <pre>yolov5mObjects365.onnx  #Modelo comprimido (pesos y arquitectura)</pre>
    - <pre>objects365.names   #Detalla clas clases del dataset </pre>
  - Dataset trained (train/test)
    - <pre>best.onnx      #Modelo comprimido (pesos y arquitectura)</pre>
  - Dataset trained (train+test/test)
    - <pre>best2.onnx      #Modelo comprimido (pesos y arquitectura)</pre>

*Cabe resaltar que para los ultimos modelos preentrenados (best.onnx y best2.onnx) se utiliza el mismo archivo de clases que en el caso del clasificador, el cual se encuentra fuera de la carpeta models con el nombre de "yololabels.txt"

Además se incluyeron 2 modelos con sus respectivos archivos de clases que resultaron del entrenamiento del clasificador:
- model.h5 y model.names #capaz de clasificar los peces de NatureConservancy
- modelX.h5 y modelX.names #capaz de clasificar los peces de los 3 datasets

</details>

<details open>
<summary>Kfolds</summary>

Contiene los resultados obtenidos del experimento del capitulo 5.4.1 dentro del documento de tesis. En ese sentido, para cada uno de las iteraciones se tiene un grafico de precisión a lo largo de las épocas entrenadas, la perdida obtenida y un archivo .txt donde se resumen las matrices de confusiones de entrenamiento, validacion y al final se obtienen los siguientes resultados: 
- Perdida final de entrenamiento
- precision final de entrenamiento
- Perdida final de validacion
- precision final de entrenamiento
Estos fueron usados para la creación de la tabla del experimento del capitulo 5.4.1.

</details>

<details open>
<summary>results</summary>

Dentro de esta carpeta se tienen los resultados de las pruebas realizadas para cada uno de los datasets utilizados para el entrenamiento del clasificador. Estas pruebas estan evidenciadas en el capitulo 5.3, donde se realizan las pruebas para la elección del clasificador. Dentro se tienen los siguientes archivos:

- [FishSpecies/NatureConservancy]
  - <pre>general_graphs      #gráficos resumen de todos los modelos </pre>
    - accuracy.png  
    - loss.png 
  - <pre> graphs             #gráficos por cada uno de los modelos </pre>
    - {accuracy/loss}{nombre_del_modelo}.png
  - <pre>info            # información del entrenamiento de cada modelo</pre>
    - <pre>logs_{nombre_del_modelo}.csv         #mismos datos que en kfolds</pre>
    - <pre>test_info_{nombre_del_modelo}.csv  #matriz de confusion y datos finales</pre>

</details>

<details open>
<summary>testingPipeline</summary>

Contiene las pruebas del pipeline finales, que se evidencian en el documento en el capítulo 5.5. Para ello, se tiene la carpeta "images", la cual contiene un 20% de la mezcla del dataset de entrenamiento y prueba de la carpeta "NatureConservancyCropped". Estas imágenes fueron procesadas por los diferentes detectores (UniDet, yolov5 y yolov5-trained) y colocadas en las demás carpetas para luego ser procesadas por el clasificador para verificar su precisión con cada benchmark.

</details>

<details open>
<summary>utils</summary>

Contiene tres archivos utilitarios para la realización de la tesis, sobreotdo para el procesamiento en tiempo real.

</details>

</details>

<details open>
<summary>Experimentos</summary>

<details open>
<summary>Experimentación del detector</summary>

- Se obtiene el 20\% del dataset combinado de training+testing desde la carpeta NatureConservancyCropped.
```
python create_testing.py
```
El comando creará la carpeta ./testingPipeline/images con las imágenes escogidas de manera aleatoria. Las imágenes sacadas del dataset de testing, serán etiquetadas como "missing".

- Por cada uno de los modelos obtenidos, se obtienen las detecciones en las imágenes y se recortan las subimágenes para crear el benchmark.

```
yolov5 preentrenado>> python detect_crop_images.py -w ./models/yolov5mObjects365.onnx -cl ./models/yololabels.txt -out yolo

yolov5 entrenado>> python detect_crop_images.py -w ./models/best.onnx -cl ./yololabels.txt -out yoloTrained3

UniDet preentrenado>> Revisar el repositorio de UniDet mencionado al inicio del documento.

```

</details>


<details open>
<summary>Experimentación para la elección del clasificador</summary>

### Experimento #1 

Se experimentó con el dataset FishSpecies para comparar el comportamiento de cada uno de los clasificadores con un dataset sencillo.

- Se entrena cada modelo con el dataset FishSpecies y se grafican los valores de precision y pérdida para cada uno de los modelos. 
```
python main.py
```
Al ejecutarse, en la carpeta ./results se van a almacenar los resultados obtenidos para cada una de las pruebas. 

*cabe resaltar que para ejecutar este comando con el dataset de FishSpecies, es necesario descomentar las lineas 317 y 325 del código.

### Experimento #2

Se experimentó con el dataset The Nature Conservancy para comparar el comportamiento de cada uno de los clasificadores con un dataset complejo y el que obtuviese el mejor resultado de manera ponderada, sería escogido para el pipeline final.

- Se entrena cada modelo con el dataset The Nature Conservancy y se grafican los valores de precision y pérdida para cada uno de los modelos. 
</details>

<details open>
<summary>Experimentación del clasificador</summary>

### Experimento #1 

Una vez se obtuvo que el modelo que mejor se comportaba fue el EfficientNetB0, se procedió a comprobar que el comportamiento no fuese alterable por la aleatoriedad:

- Se utilizó el algoritmo de KFolds sin sustitución con K=10 
```
python kfolds.py
```
Al ejecutarse, en la carpeta ./kfolds se van a almacenar los resultados obtenidos para cada una de las iteraciones. 

### Experimento #2

Con el anterior experimento se comprobó que el modelo respondía de manera correcta y constante a lo largo de todas las iteraciones, por lo que podemos afirmar que es estable para el uso propuesto. Se realizó el entrenamiento y prueba del modelo utilizando una distribución del 70-20-10% de los datos para entrenamiento, validación y prueba.

- Se ejecuta el archivo para el entrenamiento y prueba del modelo con el dataset combinado de entrenamiento. 

```
python main4_merged.py
```
Al ejecutarse, se crea el archivo "experimento2_conf_matrix.csv" en la carpeta principal, donde se almacena la matriz de confusión obtenida, además de hacer un print de las métricas obtenidas para su ejecución.

*cabe resaltar que se deben descomentar las lineas 144 y 195 para su ejecución, ya que también es usado para el experimento #3

### Experimento #3

Al comprobarse que si respondía de manera correcta al dataset objetivo con la salvedad que utilizaba únicamente el conjunto de entrenamiento, pero que tenía un problema de distribución de datos entre entrenamiento y prueba (mencionado en los anexos), se procedió a realizar la mezcla de los dos conjuntos y realizar la misma distribución anterior para entrenar el modelo con todas las imágenes.

- Se ejecuta el archivo para el entrenamiento y prueba del modelo con el dataset combinado de entrenamiento y prueba. 

```
python main4_merged.py
```
Al ejecutarse, se crea el archivo "experimento3_conf_matrix.csv" en la carpeta principal, donde se almacena la matriz de confusión obtenida, además de hacer un print de las métricas obtenidas para su ejecución.

*Este programa almacena el modelo entrenado en el archivo "model.h5" en la carpeta principal.

**Para la sustentacion y presentación de la tesis, se utilizaron tambien las clases dentro de la carpeta NatureConservancyCroppedMerged, que contenia las especies de FishSpecies. En ese sentido, se añadieron las columnas correspondientes y se genero un nuevo modelo "modelX.h5" conteniendo el nuevo modelo capaz de detectar 4 especies extra de peces.

</details>

<details open>
<summary>Experimentación del pipeline final</summary>

Una vez obtenido el clasificador del anterior experimento y obtenido los recortes de las imágenes del dataset original (en el primer experimento). Se procedió a evaluar que tan bien se comportaba el clasificador con los recortes hechos por los detectores. 

```
python main5_testing.py --dataset UniDet
python main5_testing.py --dataset yolo
python main5_testing.py --dataset yoloTrained3
```

Este programa utilizará el modelo entrenado en el experimento anterior y lo utilizará para testear el modelo con las imágenes recortadas por cada detector por separado. 

Por cada una de las ejecuciones del programa se crearán 3 archivos:
- {modelo}_conf_matrix.csv: almacena la matriz de confusión como una tabla
- {modelo}_conf_matrix.jpg: almacena la matriz de confusión como un diagrama de calor
- {modelo}_info.txt: almacena las métricas de cada experimento por separado (en términos de clasificador dentro del pipeline). Este resultado pasará a ser multiplicado con la precisión del detector que generó al recortar las imágenes.

</details>

<details open>
<summary>Anexos y archivos no utilizados</summary>

Algunos de los archivos/programas no fueron explicados dentro de esta documentación por motivos de espacio y tamaño. Aún así, hubieron experimentos que fueron importantes para el desarrollo de esta tesis. 

- data_augmentation.py: Utilizado para realizar un proceso de aumentación de las imágenes del dataset debido a que se creía que el mal rendimiento del clasificador se debía a una heterogeneidad en el número de imágenes por clase. 
- cropper.py: Archivo utilizado para realizar los recortes de las imágenes del dataset utilizando los bounding boxes creados por el software YoloLabel usado por el experto.
- main3_augmented: Utilizado para el entrenamiento de la data generada por data_augmentation.py. 

entre otros ...
</details>

</details>


<details open>
<summary>Productos finales</summary>

### Pipeline ejecutado a una imagen:

```
yolov5 pretrained >> python detect_classify.py -i ./yololabel/bet.jpg -wD ./models/yolov5mObjects365.onnx -clD ./models/objects365.names -wC ./models/modelX.h5 -clC ./models/modelX.names -expand yes

yolov5 trained >> python detect_classify.py -i ./yololabel/bet.jpg -wD ./models/best.onnx -clD ./models/yololabels.txt -wC ./models/modelX.h5 -clC ./models/modelX.names -expand no  

```

*No se realizo un pipeline para UniDet debido a 2 razones: el modelo solo está disponible para utilizar en Ubuntu y Mac y mis computadores disponibles no han sido configurados para el uso de tarjetas gráficas en ese sistema operativo y debido a que no es posible exportar el modelo a un formato estandar (onnx) para poder incluirlo dentro del mismo pipeline y ejecutarlo de manera similar.

**Cabe resaltar que UniDet utilizado sin tarjeta gráfica demora entre 2 y 3 segundos por imagen.

***Estas mismas consideraciones aplican para el siguiente producto

### Pipeline ejecutado en tiempo real con camara:

```
yolov5 pretrained >> python detect_classify_camera.py -wD ./models/yolov5mObjects365.onnx -clD ./models/objects365.names -wC ./models/modelX.h5 -clC ./models/modelX.names -expand yes

yolov5 trained >> python detect_classify_camera.py -wD ./models/best.onnx -clD ./yololabels.txt -wC ./models/modelX.h5 -clC ./models/modelX.names -expand no  

```

</details>