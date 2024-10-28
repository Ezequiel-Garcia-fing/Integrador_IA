import librosa
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import os
import matplotlib.pyplot as plt

# ----------------------------- FUNCIONES AUXILIARES -----------------------------

# Extraer características MFCC de una muestra de voz
def extraer_caracteristicas_voz(archivo_audio):
    audio_data, sample_rate = librosa.load(archivo_audio)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Cargar y mostrar una imagen correspondiente a la verdura reconocida
def mostrar_imagen(verdura):
    ruta_imagenes = 'base_datos_imagenes/'
    for archivo in os.listdir(ruta_imagenes):
        if archivo.startswith(verdura):  # Buscar la imagen que corresponda a la verdura
            imagen = cv2.imread(os.path.join(ruta_imagenes, archivo))
            # Mostrar la imagen usando matplotlib
            plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            plt.title(f"Imagen de {verdura}")
            plt.axis('off')
            plt.show()
            break

# -------------------------- BASES DE DATOS (ENTRENAMIENTO) -----------------------

# Cargar base de datos de voz y extraer características
def cargar_base_datos_voz():
    base_datos_voz = []
    etiquetas_voz = []
    ruta_voz = 'base_datos_voz/'

    for archivo in os.listdir(ruta_voz):
        if archivo.endswith('.wav'):
            caracteristicas = extraer_caracteristicas_voz(os.path.join(ruta_voz, archivo))
            base_datos_voz.append(caracteristicas)
            etiquetas_voz.append(archivo.split('_')[0])  # Etiqueta basada en el nombre del archivo (papa, zanahoria, etc.)

    return np.array(base_datos_voz), np.array(etiquetas_voz)

# -------------------------- ENTRENAMIENTO DE ALGORITMOS --------------------------

# Entrenar algoritmo Knn para reconocimiento de voz
def entrenar_knn(base_datos_voz, etiquetas_voz):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(base_datos_voz, etiquetas_voz)
    return knn

# ------------------------------- APLICACIÓN DEL SISTEMA --------------------------

# Predicción de voz con Knn
def reconocer_voz(knn, archivo_voz):
    caracteristicas_voz = extraer_caracteristicas_voz(archivo_voz)
    prediccion = knn.predict([caracteristicas_voz])
    return prediccion[0]

# ------------------------------- PROGRAMA PRINCIPAL ------------------------------

if __name__ == "__main__":
    # Paso 1: Cargar las bases de datos
    print("Cargando bases de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz()

    # Paso 2: Entrenar el modelo Knn para reconocimiento de voz
    print("Entrenando Knn para reconocimiento de voz...")
    knn = entrenar_knn(base_datos_voz, etiquetas_voz)

    # Paso 3: Simulación del sistema

    # Simulación de reconocimiento de voz
    archivo_voz_test = 'test_voz.wav'  # Cambia por el archivo de prueba de voz
    verdura_reconocida = reconocer_voz(knn, archivo_voz_test)
    print(f"Verdura reconocida por voz: {verdura_reconocida}")

    # Mostrar la imagen correspondiente a la palabra reconocida
    mostrar_imagen(verdura_reconocida)