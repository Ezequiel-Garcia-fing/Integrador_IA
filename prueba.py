import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------- FUNCIONES AUXILIARES -----------------------------

# Extraer características MFCC de una muestra de voz con eliminación de silencios
def extraer_caracteristicas_voz(archivo_audio):
    audio_data, sample_rate = librosa.load(archivo_audio)
    
    # Recortar los silencios al inicio y al final del audio
    audio_data, _ = librosa.effects.trim(audio_data)
    
    # Extraer características MFCC
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Cargar base de datos de voz y extraer características
def cargar_base_datos_voz():
    base_datos_voz = []
    etiquetas_voz = []
    ruta_voz = 'base_datos_voz/'  # Asegúrate de que la carpeta esté en el mismo directorio

    for archivo in os.listdir(ruta_voz):
        if archivo.endswith('.wav'):
            caracteristicas = extraer_caracteristicas_voz(os.path.join(ruta_voz, archivo))
            base_datos_voz.append(caracteristicas)
            etiquetas_voz.append(archivo.split('_')[0])  # Etiqueta basada en el nombre del archivo (papa, zanahoria, etc.)

    # Convertir a array para usar en Scikit-Learn
    base_datos_voz = np.array(base_datos_voz)
    etiquetas_voz = np.array(etiquetas_voz)

    # Normalizar las características
    scaler = StandardScaler()
    base_datos_voz = scaler.fit_transform(base_datos_voz)  # Ajustar y transformar los datos de entrenamiento

    return base_datos_voz, etiquetas_voz

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
    # Cargar la base de datos de voz y sus etiquetas
    print("Cargando la base de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz()

    # Visualizar los datos con PCA
    print("Visualizando características de voz con PCA...")
    visualizar_datos(base_datos_voz, etiquetas_voz)
