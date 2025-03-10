import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb  # type: ignore
from keras import models, layers
from tensorflow.keras.utils import plot_model  # type: ignore

def Binary_Classification():

    # Cargar los datos de IMDB (reseñas de películas etiquetadas como positivas o negativas)
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Función para convertir las secuencias de palabras en vectores binarios (one-hot encoding)
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))  # Matriz de ceros
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.  # Marcar con 1 las posiciones correspondientes a las palabras en la secuencia
        return results

    # Vectorizar los datos de entrenamiento y prueba
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Convertir las etiquetas a un formato adecuado para el entrenamiento
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # Construir el modelo de red neuronal
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # Capa densa con 64 unidades y activación ReLU
    model.add(layers.Dense(16, activation='relu'))  # Otra capa densa con 16 unidades
    model.add(layers.Dense(1, activation='sigmoid'))  # Capa de salida con activación sigmoide (para clasificación binaria)

    # Compilar el modelo: definir el optimizador, la función de pérdida y las métricas
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',  # Función de pérdida para clasificación binaria
                  metrics=['accuracy'])  # Métrica: precisión

    # Dividir los datos de entrenamiento en entrenamiento y validación
    x_val = x_train[:10000]  # Datos de validación (primeras 10,000 muestras)
    partial_x_train = x_train[10000:]  # Datos de entrenamiento (resto de las muestras)
    y_val = y_train[:10000]  # Etiquetas de validación
    partial_y_train = y_train[10000:]  # Etiquetas de entrenamiento

    # Entrenar el modelo
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,  # Número de épocas
                        batch_size=512,  # Tamaño del lote
                        validation_data=(x_val, y_val))  # Datos de validación

    # Obtener los datos del historial de entrenamiento
    history_dict = history.history
    loss_values = history_dict['loss']  # Pérdida en entrenamiento
    val_loss_values = history_dict['val_loss']  # Pérdida en validación
    acc_values = history_dict['accuracy']  # Precisión en entrenamiento
    val_acc_values = history_dict['val_accuracy']  # Precisión en validación
    epochs = range(1, len(loss_values) + 1)  # Número de épocas

    # Crear una figura con dos subgráficos para visualizar el rendimiento del modelo
    plt.figure(figsize=(12, 5))

    # Subgráfico 1: Pérdida (loss) durante el entrenamiento y validación
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')  # Pérdida en entrenamiento
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # Pérdida en validación
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Subgráfico 2: Precisión (accuracy) durante el entrenamiento y validación
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')  # Precisión en entrenamiento
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')  # Precisión en validación
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Mostrar la figura
    plt.tight_layout()
    plt.show()

    # Evaluar el modelo en el conjunto de prueba
    results = model.evaluate(x_test, y_test)
    print("Test Loss:", results[0])  # Pérdida en el conjunto de prueba
    print("Test Accuracy:", results[1])  # Precisión en el conjunto de prueba

    # Realizar una predicción con el modelo entrenado
    print("Predicción para las dos primeras muestras del conjunto de prueba:")
    print(model.predict(x_test[0:2]))  # Predicción para las dos primeras muestras

