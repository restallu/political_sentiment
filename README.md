# <font color=gree>LLM. Detección de sentimiento político en un texto</font>
En este repositorio desplegamos una app con el objetivo de detectar la tendencia política de un texto: izquierda o derecha
Forma parte del proyecto fin de master del master en gestión y análisis de grandes volúmenes de datos: BIG DATA de la Universidad
Europea Miguel de Cervantes.

En este  proyecto se han probado varios LLM y al final se  ha usado para la predicción el que mejor resultados obtenía que en nuestro
caso ha sido el modelo RoBERTa entrenado en español con un corpus de datos procedentes de la Biblioteca Nacional:
'PlanTL-GOB-ES/roberta-base-bne'

### <font color=gree>Datos de Entrenamiento</font>

La aplicación se ha probado con datos de entrenamiento procedentes de dos fuentes:
   1. Sesiones de investidura del Congreso de los Diputados de España
   2. Artículos de opinión de diaros digitales con marcada tendencia política   

El etiquetado de los datos ha sido de tipo binario (0-> Izquierda y 1-> Derecha). No se han etiquetado otras tipologías para mantener el mayor grado posible de objetividad

### <font color=gree>Tratamiento de los datos</font>

El tamaño inicial de los datos que aliementan el sistema puede variar entre varios cientos de palabras  en el caso de artículos cortos hasta varias decenas de miles en el caso de los discursos políticos. Por eso se ha dividido inicialmente los datos en paquetes (ficheros) de 500 palabras cada uno (aprox 3Kb). De esta manera resulta una cifra de algo más de 1000 documentos siendo la relacion izquierda derecha del orden  de 51% 49 % resultando así un set de pruebas equilibrado.

Sobre este conjunto de datos se aplican dos tipos de procesamientos previos al LLM:
1. Se pasan a minusculas y se eliminan signos de puntuación
2. Se procesan segun lo dicho en 1., y además se le eliminan las palabras de uso común (stop words)

Para cada uno de estos sets de datos se les  aplica 4 tipos de algoritmos:

1. BERT más capa neuronal de clasificación a la salida
2. distilBERT más capa clasificación integrada 
3. RoBERTa más capa de clasificación integrada
4. GPT2 mas capa de clasificación

Y cada simulacion de las anteriores se realiza para 2 tamaños de secuencia de entrada: 125 y 250

El resumen de resultados para el valor de accuracy el siguiente:

Caso 1 (valores de accuracy para seq entrada sin procesamiento)

|Modelo/secuencia|125|250|
|----------------|----|---|
|BertClassifier|0.8745|0.8524|
|RobertaClassifier|0.9225|**0.9041**|
|DistilbertClassifier|0.8745|0.8635|
|GPT2Classifier|0.7259|0.7269|

Caso 2 (valores de accuracy para seq de entrada con procesamiento)

|Modelo/secuencia|125|250|
|----------------|----|---|
|BertClassifier|0.8798|0.8708|
|RobertaClassifier|0.8966|**0.9041**|
|DistilbertClassifier|0.8708|0.8524|
|GPT2Classifier|0.7138|0.7823|


  * Todos los datos son correspondientes al dataset de test

A la vista de los datos anteriores hemos desplegado en github el caso 1 de RoBERTa como clasificador. Esto ha sido así ya que ofrece el mejor valor y (aunque mejora en el caso 2) hemos notado un comportamiento peor en general y más inestable en el caso de preproceso de los datos tipo 2.

## <font color=gree>Operativa </font>

La operativa es simple. Podemos o bien copiar y pegar un texto o bien subir un fichero de texto y pulsar submit. En el caso de querer introducir un texto largo (por ejemplo artículo de opinión) lo subiremos como fichero y el sistema lo dividirá en trozos de 250 palabras. Si el remanente final de uno de los trozos es menor de 50 palabras (20% de tamaño máximo) se descarta. Si el resultado de los trozos no es el mismo se selecciona el valor que se repita más veces.

[README](https://github.com/restallu/political_sentiment/blob/main/README.md)