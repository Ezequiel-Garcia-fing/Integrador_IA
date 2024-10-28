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

def extraer_caracteristicas_completas(archivo_audio):
    try:
        audio_data, sample_rate = librosa.load(archivo_audio)
        audio_data, _ = librosa.effects.trim(audio_data)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=22)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        caracteristicas = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean])
        return caracteristicas
    except Exception as e:
        print(f"Error al extraer características: {e}")
        return None

def cargar_base_datos_voz_completa():
    base_datos_voz = []
    etiquetas_voz = []
    ruta_voz = 'base_datos_voz/'
    for archivo in os.listdir(ruta_voz):
        if archivo.endswith('.wav'):
            caracteristicas = extraer_caracteristicas_completas(os.path.join(ruta_voz, archivo))
            if caracteristicas is not None:
                base_datos_voz.append(caracteristicas)
                etiquetas_voz.append(archivo.split('_')[0])
    base_datos_voz = np.array(base_datos_voz)
    etiquetas_voz = np.array(etiquetas_voz)
    scaler = StandardScaler()
    base_datos_voz = scaler.fit_transform(base_datos_voz)
    return base_datos_voz, etiquetas_voz

def mostrar_imagen(verdura):
    try:
        ruta_imagenes = 'base_datos_imagenes/'
        for archivo in os.listdir(ruta_imagenes):
            if archivo.startswith(verdura):
                imagen = cv2.imread(os.path.join(ruta_imagenes, archivo))
                plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                plt.title(f"Imagen de {verdura}")
                plt.axis('off')
                plt.show()
                break
    except Exception as e:
        print(f"Error al mostrar imagen: {e}")

def entrenar_knn(base_datos_voz, etiquetas_voz):
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(base_datos_voz, etiquetas_voz)
    return knn

def reconocer_voz(knn, archivo_voz):
    caracteristicas_voz = extraer_caracteristicas_completas(archivo_voz)
    if caracteristicas_voz is not None:
        prediccion = knn.predict([caracteristicas_voz])
        return prediccion[0]
    else:
        return None

def grabar_audio_desde_microfono():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Por favor, di el nombre de una verdura...")
        audio = recognizer.listen(source)
        with open("temp_voz.wav", "wb") as f:
            f.write(audio.get_wav_data())
    return 'temp_voz.wav'

def visualizar_datos(base_datos_voz, etiquetas_voz):
    pca = PCA(n_components=2)
    reduccion = pca.fit_transform(base_datos_voz)
    etiquetas_unicas = list(set(etiquetas_voz))
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
    print("Cargando bases de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz_completa()
    print("Entrenando Knn para reconocimiento de voz...")
    knn = entrenar_knn(base_datos_voz, etiquetas_voz)
    print("Visualizando características de voz con PCA...")
    visualizar_datos(base_datos_voz, etiquetas_voz)
    archivo_voz_test = grabar_audio_desde_microfono()
    verdura_reconocida = reconocer_voz(knn, archivo_voz_test)
    if verdura_reconocida:
        print(f"Verdura reconocida por voz: {verdura_reconocida}")
        mostrar_imagen(verdura_reconocida)
    else:
        print("No se pudo reconocer la verdura.")
