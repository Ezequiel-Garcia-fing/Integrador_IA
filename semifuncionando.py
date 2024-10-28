import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import speech_recognition as sr
import cv2

# ----------------------------- FUNCIONES AUXILIARES -----------------------------

# Extraer múltiples características de una muestra de voz (MFCC, Chroma, Spectral Contrast)
def extraer_caracteristicas_completas(archivo_audio):
    audio_data, sample_rate = librosa.load(archivo_audio)
    
    # Recortar silencios
    audio_data, _ = librosa.effects.trim(audio_data)

    # MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Chroma (Croma del espectrograma)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Spectral Contrast (Contraste espectral)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    # Concatenar todas las características
    caracteristicas = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean])
    
    return caracteristicas

# Cargar base de datos de voz y extraer características
def cargar_base_datos_voz_completa():
    base_datos_voz = []
    etiquetas_voz = []
    ruta_voz = 'base_datos_voz/'

    for archivo in os.listdir(ruta_voz):
        if archivo.endswith('.wav'):
            caracteristicas = extraer_caracteristicas_completas(os.path.join(ruta_voz, archivo))
            base_datos_voz.append(caracteristicas)
            etiquetas_voz.append(archivo.split('_')[0])  # Etiqueta basada en el nombre del archivo (papa, zanahoria, etc.)

    # Convertir a array para usar en Scikit-Learn
    base_datos_voz = np.array(base_datos_voz)
    etiquetas_voz = np.array(etiquetas_voz)

    # Normalizar las características
    scaler = StandardScaler()
    base_datos_voz = scaler.fit_transform(base_datos_voz)  # Ajustar y transformar los datos de entrenamiento

    return base_datos_voz, etiquetas_voz

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

# Entrenar algoritmo Knn para reconocimiento de voz
def entrenar_knn(base_datos_voz, etiquetas_voz):
    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(base_datos_voz, etiquetas_voz)
    return knn

# Predicción de voz con Knn
def reconocer_voz(knn, archivo_voz):
    caracteristicas_voz = extraer_caracteristicas_completas(archivo_voz)
    prediccion = knn.predict([caracteristicas_voz])
    return prediccion[0]

# Capturar audio desde micrófono
def grabar_audio_desde_microfono():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Por favor, di el nombre de una verdura...")
        audio = recognizer.listen(source)

        # Guardar el audio capturado como archivo .wav temporal
        with open("temp_voz.wav", "wb") as f:
            f.write(audio.get_wav_data())

    return 'temp_voz.wav'

# Visualizar las características en 2D usando PCA
def visualizar_datos(base_datos_voz, etiquetas_voz):
    pca = PCA(n_components=2)  # Reducimos las dimensiones a 2D para visualizar
    reduccion = pca.fit_transform(base_datos_voz)

    # Convertir etiquetas a números (por ejemplo, 'papa' -> 0, 'zanahoria' -> 1)
    etiquetas_unicas = list(set(etiquetas_voz))  # Obtener etiquetas únicas
    etiquetas_numericas = [etiquetas_unicas.index(label) for label in etiquetas_voz]

    plt.scatter(reduccion[:, 0], reduccion[:, 1], c=etiquetas_numericas, cmap='viridis')
    plt.title('Visualización PCA de las características de voz')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.colorbar(ticks=range(len(etiquetas_unicas)), label='Clase de verdura')
    plt.clim(-0.5, len(etiquetas_unicas) - 0.5)
    plt.show()

# ------------------------------- PROGRAMA PRINCIPAL ------------------------------

if __name__ == "__main__":
    # Paso 1: Cargar las bases de datos
    print("Cargando bases de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz_completa()

    # Paso 2: Entrenar el modelo Knn para reconocimiento de voz
    print("Entrenando Knn para reconocimiento de voz...")
    knn = entrenar_knn(base_datos_voz, etiquetas_voz)

    # Visualizar los datos con PCA
    print("Visualizando características de voz con PCA...")
    visualizar_datos(base_datos_voz, etiquetas_voz)

    # Paso 3: Capturar audio directamente desde el micrófono
    archivo_voz_test = grabar_audio_desde_microfono()  # Graba el audio desde el micrófono

    # Paso 4: Reconocer la verdura con el modelo entrenado
    verdura_reconocida = reconocer_voz(knn, archivo_voz_test)
    print(f"Verdura reconocida por voz: {verdura_reconocida}")

    # Paso 5: Mostrar la imagen correspondiente a la palabra reconocida
    mostrar_imagen(verdura_reconocida)
