# 💡 Binary Classification NLP 💡
**Estudiante:** Aarón Enrique Ramírez González  

**Tarea:** Clasificación Binaria  

**Materia:** Sistemas de Visión Artificial  

---

## 📝 Descripción
Este repositorio contiene la implementación de una red neuronal para clasificación binaria utilizando el conjunto de datos IMDB (reseñas de películas etiquetadas como positivas o negativas). 

El proyecto incluye:

- 🧩 **Carga de datos:** Carga y preprocesamiento del conjunto de datos IMDB.
- 🛠️ **Construcción de la red neuronal:** Implementación de una red neuronal con capas densas y funciones de activación ReLU y sigmoide.
- 🚀 **Entrenamiento y evaluación:** Entrenamiento del modelo y visualización de métricas de rendimiento (pérdida y precisión).
- 📊 **Visualización:** Gráficas de pérdida y precisión durante el entrenamiento y la validación.


---

## 📋 Requisitos
Para ejecutar este proyecto, necesitas tener instaladas las siguientes dependencias:

- **Python:** En este caso, la versión utilizada es la **3.12.4**.
- **NumPy:** Para cálculos numéricos y manejo de arreglos.
- **Matplotlib:** Para la generación de gráficas.
- **TensorFlow y Keras:** Para la construcción y entrenamiento de la red neuronal.

Puedes instalar estas dependencias utilizando `pip`:

```bash
pip install numpy matplotlib tensorflow
```
## 🗂️ Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

``` bash
TAREA_5/
│
├── src/
│   └── Binary_Classification_NLP.py  # Script principal de la red neuronal
│
├── .gitignore      # Archivo para ignorar archivos no deseados
├── main.py         # Script principal para ejecutar el proyecto
├── README.md       # Este archivo
└── Requirements.txt # Lista de dependencias del proyecto
```
## 🚀 ¿Cómo usar este repositorio?
Sigue estos pasos para ejecutar el proyecto en tu equipo:

### Clona el repositorio 🖥️:
Abre una terminal y ejecuta el siguiente comando para clonar el repositorio en tu computadora:

```bash
git clone https://github.com/Aaron040-Rmz/Binary_Classification_NLP_2230294
```
### Crea un nuevo entorno virtual:
Se recomienda crear un entorno virtual en la carpeta principal para un fácil acceso. Su activación y desactivación se realiza de la siguiente forma:

En PowerShell:

```bash
.\nombre_del_entorno\Scripts\Activate
deactivate
```
En Unix:

```bash
source nombre_del_entorno/bin/activate
deactivate
```
### Instala las dependencias 📦:
Asegúrate de tener instaladas las bibliotecas necesarias. Ejecuta el siguiente comando para instalarlas:

```bash
pip install -r Requirements.txt
```
### Ejecuta el script principal 🚀:
Para entrenar y evaluar la red neuronal, ejecuta:

```bash
python main.py
```
### Visualiza los resultados 📊:
El script mostrará el error durante el entrenamiento en la consola.
También se mostrará un gráfico con la pérdida y precisión durante el entrenamiento y la validación. 

## 🛠️ Tecnologías Utilizadas
- **Python**: Lenguaje de programación principal (versión 3.12.4).

- **NumPy**: Para cálculos numéricos y manejo de arreglos.

- **Matplotlib**: Para visualización de datos y gráficos.

- **TensorFlow y Keras**: Para la construcción, entrenamiento y evaluación de la red neuronal.

## 🧑‍💻 ¿Qué hace el código?
El código realiza lo siguiente:

1. Carga de datos:

  * Carga el conjunto de datos IMDB, que contiene reseñas de películas etiquetadas como positivas (1) o negativas (0).

  * Convierte las secuencias de palabras en vectores binarios (one-hot encoding).

2. Construcción del modelo:

  * Define una red neuronal con dos capas ocultas (64 y 16 unidades) y una capa de salida con activación sigmoide.

  * Compila el modelo utilizando el optimizador RMSprop, la función de pérdida binary_crossentropy y la métrica de precisión.

3. Entrenamiento y evaluación:

  * Entrena el modelo durante 20 épocas y valida su rendimiento en un conjunto de validación.

  * Muestra gráficas de pérdida y precisión durante el entrenamiento y la validación.

  * Evalúa el modelo en el conjunto de prueba y realiza predicciones para las primeras muestras.

4. Visualización:

  * Genera gráficas de pérdida y precisión para analizar el rendimiento del modelo.

## 🙏 Agradecimientos
¡Gracias por ver mi "readme"! 😃

Como lo hice en los anteriores repositorios, te deseo un muy buen día y que Dios te bendiga en gran manera a ti y a toda tu familia. 😊