# Proyecto-final-Inteligencia-Artficial

                                                 I.	INTRODUCCION AL PROYECTO

El siguiente proyecto tiene como objetivo la clasificación y predicción de ataques cardiacos en una persona teniendo en cuenta distintas características, para ello, se realizan distintos pasos y se cumplen distintos parámetros para obtener el resultado esperado. Se realizan métodos de machine learning de tipo supervisado, en donde se emplean soluciones con máquinas de soporte vectorial utilizando PCA, clasificadores Ovo y ova, k vecinos más cercanos y por último maquinas de soporte con kernel RBF. Para realizar este proyecto se hizo uso de la herramienta de trabajo GOOGLE COLAB, la cual es un entorno de colaborativo de Google que permite trabajar con Notebooks y el lenguaje de programación Python.

Ahora bien, para poder realizar este proyecto fue necesario implementar un conjunto de datos que se adecuará al objetivo del proyecto, para realizar esto, se determino por seguir un paso a paso, con ello, ya se procede a definir qué conjunto de datos usar y como implementarlos, los pasos a seguir se presentan a continuación.

•	¿Cuántos datos tiene, y son continuos o categóricos?

•	¿El problema está relacionado con la clasificación, asociación, agrupación o regresión?

•	¿Variables predefinidas (etiquetadas), sin etiquetar o mixtas?

•	¿Cuál es el objetivo?

Este paso se hizo para poder tener una mayor claridad frente al conjunto de datos a utilizar, en este caso, se obtuvo un dataset suministrado de Kaggle, el cual se denomina “Heart” [1], este dataset contiene distintas características a tener en cuenta, también, presenta todos los datos categóricos, por lo que, no fue necesario hacer una curación de datos, ya que los datos se presentaban de manera óptima, no obstante, se verifico si el conjunto de datos con respecto al problema se igualaban al método para aplicar, dando como resultado un método de clasificación y de regresión, finalmente, se obtuvieron variables predefinidas, es decir, etiquetadas para que con ello se pueda cumplir con el método de aprendizaje supervisado.

Para poder entender mejor el conjunto de datos a utilizar, se presentan cada una de sus características que genera un maro conocimiento para poder implementarlo.

•	Edad: Edad del paciente.

•	Sexo: Sexo del paciente.

•	exang: angina inducida por el ejercicio (1 = sí; 0 = no).

•	ca: número de buques principales (0-3).

•	cp : Tipo de dolor torácico tipo de dolor torácico.

•	Valor 1: angina típica.

•	Valor 2: angina atípica.

•	Valor 3: dolor no anginoso.

•	Valor 4: asintomático.

•	trtbps: presión arterial en reposo (en mm Hg).

•	chol: colestoral en mg/dl obtenido a través del sensor BMI.

•	fbs: (azúcar en sangre en ayunas > 120 mg/dl) (1 = verdadero; 0 = falso).

•	rest_ecg : resultados electrocardiográficos en reposo.

•	Valor 0: normal.

•	Valor 1: tener anomalías en la onda ST-T (inversiones de la onda T y/o elevación o depresión del ST > 0,05 mV).

•	Valor 2: mostrar hipertrofia ventricular izquierda probable o definitiva según los criterios de Estrés.

•	thalach: frecuencia cardíaca máxima alcanzada.

•	objetivo: 0= menos posibilidades de ataque al corazón.  1= más posibilidades de ataque al corazón.


                                                      II.	OBJETIVO GENERAL.

Realizar la clasificación y predicción de ataques cardiacos teniendo en cuenta un conjunto de datos con características específicas

                                                  III.	OBJETIVOS ESPECIFICOS.

o	Realizar la búsqueda de la base de datos con los parámetros adecuados para obtener la clasificación y predicción de los ataques cardiacos.

o	Ordenar y realizar el análisis de la base de datos que permita estimar el alcance y los límites del proyecto.

o	Plantear los métodos de clasificación que se aplicaron a los datos y que pueden permitir obtener el mínimo de error en la estimación de la experiencia.

o	Concluir acerca de la efectividad del modelo para estimar la clasificación y predicción de ataques al corazón.

Ya con los parámetros especificados, se procede a realizar el tratamiento de datos realizados en colab, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203171919-c2c73bd5-ef77-4c94-8fa5-9ecbf28cf0ab.png)

Luego, se obtiene el histograma de las características, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182127-431d0a9b-19a3-4fd5-a8b3-13943b9c9473.png)

                                            IV.	SECCION DE DESARROLLO.

Para iniciar, es necesario mencionar que, para este caso, dado el dataset que se utilizó, no hubo presencia de datos de tipo NaN, por ello, no hubo necesidad de realizar una curación de datos, o hablando mas especifico, no se rellenaron las casillas con otros datos. Con esto claro, se procede a realizar una grafica donde se pueda obtener un histograma, el cual se presenta en la primera parte de este documento, con este histograma se puede determinar el comportamiento de estos mismos datos.

También, es importante aclarar que debido a la clasificación y predicción a dicho atributo, de inicio, se escalizan los datos, luego, a estos datos escalizados se les realiza PCA, pero para este caso se tenían 12 componentes, por lo que, al realizaron se obtuvieron 10 componentes, esto presentaba una varianza explicada que seria menor a un 97%, lo que genera una gran pérdida de información, un 3% o quizá más porcentaje de perdida, por esta razón, no se realiza a nivel general una reducción dimensional.

Teniendo esto en cuenta, los métodos que se implementaron fueron:

•	Algoritmo K vecinos mas cercanos: Se hace uso del algoritmo de knn para obtener una respuesta acertada, este algoritmo tiene algunas consideraciones, una de ellas es que solo es necesaria 1 muestra por clase para un clasificador, lo que se aplico en el proyecto, además, se debe seleccionar cuidadosamente el k, ya que con ello varían mucho los hiperparametros, además, se tuvo en cuenta los problemas que esta solución puede conllevar, los cuales son:

i.	Si se toman muchas muestras X, la clasificación pude llegar a ser muy lenta, esto, no beneficia a ningún conjunto de datos.

ii.	No existe un método estándar para determinar un valor óptimo para K.

iii.	Los valores pequeños de K son mas susceptibles a afectaciones por valores fuera de tendencia de ruido.

iv.	Los valores grandes de K son más inmunes a ruido, pero si K es muy grande las categorías con pocas muestras pueden llegar a no ser seleccionadas nunca.

v.	Si se selecciona mal el K se puede llegar fácilmente a grandes regiones incongruentes empatadas en votación. 

•	Maquinas de soporte vectorial con kernel RBF: Se implemento este método en donde se vario cada uno de sus hiperparametros utilizando (For), también se hizo por métodos un poco mas eficientes como lo son GridSearch.

•	OneVsOneClassifier: se implementó este método en donde se pudo determinar el óptimo “K” para que este método cumpliera con las mejores métricas, como, por ejemplo, el coeficiente de correlación de matthews y el accuracy, No obstante, se tuvieron en cuenta las desventajas de este método, las cuales son.

i.	La escala de los valores de confianza puede diferir entre los clasificadores binarios.

ii.	El numero de clasificadores aumenta con respecto a OvA pero son clasificadores mas simples para un K grande.
iii.	Dependiendo de la implementación pueden llegarse a dar soluciones ambiguas.

Con respecto al uso de PCA en el presente problema, se presentan a continuación los resultados obtenidos al realizar este modelo,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182456-604c3795-4005-45e5-8ac3-5f00e391b5a0.png)

Tal como se pude observar en la imagen, se pude optar por dejar 12 componentes principales, tomando como perdida un 3% de información de los datos suministrados.

                                                    V.	RESULTADOS.

Para poder realizar el proyecto se realizaron diferentes prácticas, para iniciar se realizó la clasificación y predicción con Knn tal como se muestra en la siguiente imagen,

•	clasificación y predicción a los ataques cardiacos implementando Knn.

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182747-6c924904-6e78-468b-8589-b0801bd6eb72.png)

En este paso, se obtiene el accuracy del clasificador donde, si da un valor diferente a 1, significa que hay muestras mal etiquetadas, por ello, se prueba el conjunto que se esta entrenando. Luego, se obtiene el accuracy de la prueba con el resto de los datos que no fueron incluidos en el conjunto de entrenamiento anterior, con ello, se obtienen dos resultados esperando un valor aceptable.

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182825-b494b372-2ad3-4da0-95f3-742bf700f62a.png)

Para cada una de las pruebas y tal como se observa en la imagen dio un valor alto, dado que, se obtuvo un valor para el clasificador entrenado de 0.83, y para el resto del conjunto un valor de 0.87, lo cual demuestra un buen valor.

Luego, se realiza la predicción de Knn para obtener la matriz de confusión y con ello poder obtener el coeficiente de Matthews y el Accuracy correspondiente, esperando obtener un valor aceptado con el conjunto de entrenamiento,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182886-3299f371-4635-4fce-9715-38fba28f5a91.png)

Tal como se observa en la imagen, se obtuvo un coeficiente de matthews de 0.73 y también, un accuracy de 0.83, lo que implica una precisión muy buena referente al conjunto de entrenamiento.

Luego de obtener esta precisión, obteniendo el mejor valor de k, se procedió a realizar la gráfica de la accuracy con un rango de (1, 20) para obtener en si la siguiente gráfica,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203182971-0152ff5f-82e7-41f6-9345-370a7605f796.png)


Donde, se puede observar una precisión muy buena, teniendo como punto mínimo un valor de 0.78 y con un valor máximo de aproximadamente 0.87, lo que esclarece un resultado positivo para el conjunto de entrenamiento.

Finalmente, se obtienen el ACC y el MCC variando K, utilizando un for para sacar los scores de X_test y Y_test, tal como se muestra a continuación.

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183086-3c341309-f1c8-4a80-920e-ec391f140b49.png)

La grafica para ACC se obtiene a continuación,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183140-461de073-4ffc-48f1-9172-87a88c08fba9.png)

Del mismo modo, se obtiene la gráfica para MCC,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183203-7a8eada8-6cd1-430e-990d-fbcd89058b1a.png)

Donde, al observar las dos imágenes se obtienen resultados decentes de la precisión.

•	Clasificación para los ataques cardiacos aplicando el método One vs One (OVO).

Como segunda parte, se realizó el método de clasificación One vs One (OVO), para poder obtener resultados congruentes y con ello poder obtener un buen resultado, para iniciar, se implementaron las librerías,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183278-48ad08e0-7d6d-4e45-86c4-6bb97371b75d.png)

Luego de esto, se realiza la codificación para poder obtener la matriz de confusión, y con ello poder obtener un coeficiente de matthews y una precisión, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183320-bb9fa417-ebcd-463c-87b7-d0d3ec26e5fd.png)

Al realizar esto, se obtiene que el coeficiente de matthews y la precisión da un valor de, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183407-ddd08325-f60b-48c6-9465-028718082eda.png)

Dando de esa manera, un resultado oprimo tanto para el coeficiente como para la precisión, ya que, da un valor de 0.81 y se representa como un valor optimo.

Finalmente, se obtiene la roc de y_test, el predict, con ello obteniendo un valor de, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183513-976dc52a-e742-4b7a-a19b-49aace533eea.png)


•	clasificación y predicción de los ataques cardiacos con máquinas de soporte con Kernel rbf.

Para iniciar, se implementó kernel rbf para obtener un resultado optimo, primero, se implementan las características ya mencionadas en la primera parte, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183588-d7b1da7c-9583-447c-83dd-047d2c6d3554.png)

Luego de esto, se escalizan los datos con el conjunto de entrenamiento, el conjunto de validamiento y las muestras nuevas, tal como se observa a continuación, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183639-2d5b8cc7-323e-44e9-bb9b-5932e8cc0cbd.png)


Luego, se implementó el predict y el score para poder hallar el coeficiente de matthews y el accuracy, teniendo como resultado

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183742-fc61cae4-8a29-45db-a01f-65741f7f7c30.png)


Dando un resultado optimo, demostrando que el método de clasificación a usar es aceptable y da unos valores óptimos, ya que, la precisión da un resultado de 0.83, un valor positivo, 

Ahora, se realiza una operación con for para obtener los hiperparametros, esto variando la gama en distintos rangos, teniendo como opción que si C es muy grande no se obtendrá una regulación, por el contrario, si el resultado es pequeño entonces va a regularizar fuerte, tal como se muestra a continuación, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183790-07df217d-a86e-4ad8-acf0-05527ef1693f.png)

Ya con esto, se obtiene la gráfica del ACC variando gamma tal como se muestra a continuación, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183836-b2427682-2484-4e85-ad4a-7e66b564fcd3.png)

Finalmente, se obtiene la matriz de confusión en el conjunto de validación, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203183976-e859844a-60a8-4f61-821d-199108088f19.png)


Ahora, se realiza la clasificación y predicción con maquinas de soporte con kernel rbf aplicándole PCA.

Se realiza la codificación para obtener el coeficiente de matthews y el accuracy, tal como se muestra a continuación, 

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203184030-f68f4b3c-d310-4b05-972c-4cefc608461a.png)

Dando como resultado, una buena precisión. 

Ahora, se realiza PCA con respecto al numero de componentes y la varianza explicativa acumulada, tal como se muestra a continuación,

![Texto alternativo](https://user-images.githubusercontent.com/68627997/203184097-478585a1-43d9-4fea-8556-e2c18b6d9fa9.png)

                                                  VI.	CONCLUSIONES.
  
•	De acuerdo con los resultados que se obtuvieron en el proyecto, se puede concluir que el mejor clasificador que se implementó fue knn (K vecinos más cercanos), aplicado al problema de los ataques cardiacos, esto se deduce basándose en las métricas como el coeficiente de Matthew, el cual dio un valor aproximado a 0.73, un valor aceptable y totalmente aceptable en una escala de valores óptimos. 

•	Teniendo en cuenta que se realizó PCA a los datos, se puede deducir que no se puede hacer una reducción dimensional debido a que la varianza explicada cuando las componentes principales son menor a 12, presentan un aproximado del 95%, esto, genera que se pierda bastante información importante acerca de los datos, por esto, se realiza Pca pero no se hace reducción dimensional.

•	Se logró realizar un clasificador que pudiese predecir el problema con todo el conjunto de datos a caracterizar.

•	De acuerdo con lo realizado en este proyecto, se puede concluir que es de gran importancia que los datos estén escalizados, para este caso, se utilizó (standarscaler), esto se debe a que, muchas de las características tenían una varianza alta, esto se deduce como que presentaban una escala diferente y con mayor riesgo de dificultar el proceso. No obstante, cabe mencionar que realizar esto, facilita el poder obtener una clasificación con un buen clasificador, ya que, con ello, se puede llegar a una solución mas rápida, con ello, se converge en menos tiempo.

•	Con respecto a la parte de clasificación y predicción de los ataques al corazón, se puede afirmar que aunque no fue con mayor claridad la respuesta, se puede determinar que no se puede medir un clasificador por medio del accuracy, ay que, como se pude observar se obtuvo un accuray relativamente alto, pero, al revisar la matriz de confusion, se pudieron encontrar bastantes errores, esto teniendo en cuenta el  desempeño de cada método mediante la matriz de confusion y el coeficiente de correlación de Matthews. Por último, el mejor método fue K vecinos mas cercanos (Knn).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

LINK DE ACCESO AL VIDEO EN YOUTUBE.

[enlace en línea](https://youtu.be/wXb5mUUe9xw)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

