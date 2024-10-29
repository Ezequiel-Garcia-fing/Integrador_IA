import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import speech_recognition as sr
import cv2
import csv
from collections import defaultdict

# ----------------------------- FUNCIONES AUXILIARES -----------------------------

def extraer_caracteristicas_completas(archivo_audio, umbral_amplitud=0.029):
    audio_data, sample_rate = librosa.load(archivo_audio)
    
    # Recortar silencios
    audio_data, _ = librosa.effects.trim(audio_data)

    # Dividir el audio en 10 segmentos
    num_segmentos = 10
    duracion_segmento = len(audio_data) // num_segmentos
    segmentos = [
        audio_data[i * duracion_segmento: (i + 1) * duracion_segmento]
        for i in range(num_segmentos)
    ]
    
    # Filtrar segmentos con amplitud baja
    segmentos_filtrados = [
        seg for seg in segmentos if np.mean(np.abs(seg)) > umbral_amplitud
    ]
    
    # Si no hay segmentos válidos, advertir y devolver un vector de ceros del tamaño adecuado
    if not segmentos_filtrados:
        print(f"Advertencia: archivo {archivo_audio} no tiene segmentos con suficiente amplitud.")
        return np.zeros(7 + 12 + 7 + 1)  # Ajustar tamaño según la cantidad de características
    
    # Extraer características de cada segmento válido y calcular la media y desviación estándar
    mfccs_mean, chroma_mean, spectral_contrast_mean, zcr_mean = [], [], [], []
    for seg in segmentos_filtrados:
        # MFCC
        mfccs = librosa.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=7)
        mfccs_mean.append(np.mean(mfccs, axis=1))
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=seg, sr=sample_rate)
        chroma_mean.append(np.mean(chroma, axis=1))
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=seg, sr=sample_rate)
        spectral_contrast_mean.append(np.mean(spectral_contrast, axis=1))
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=seg)
        zcr_mean.append(np.mean(zcr))
    
    # Tomar la media y desviación estándar de las características en los segmentos válidos
    mfccs_mean = np.mean(mfccs_mean, axis=0)
    chroma_mean = np.mean(chroma_mean, axis=0)
    spectral_contrast_mean = np.mean(spectral_contrast_mean, axis=0)
    zcr_mean = np.mean(zcr_mean)
    
    # Concatenar todas las características en un solo vector
    caracteristicas = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, zcr_mean])
    
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

# Filtrar características que tengan baja desviación estándar para cada clase
def filtrar_caracteristicas_por_clase(base_datos_voz, etiquetas_voz, umbral=0.81):
    # Crear un diccionario para almacenar las características por clase
    caracteristicas_por_clase = defaultdict(list)
    
    # Agrupar características por clase (verdura)
    for i, etiqueta in enumerate(etiquetas_voz):
        caracteristicas_por_clase[etiqueta].append(base_datos_voz[i])
    
    # Convertir listas de características por clase a matrices y calcular desviación estándar
    desviaciones_std = []
    for clase, caracteristicas in caracteristicas_por_clase.items():
        matriz_caracteristicas = np.array(caracteristicas)
        desviaciones_std.append(np.std(matriz_caracteristicas, axis=0))
    
    # Calcular la media de desviaciones estándar a través de las clases
    desviaciones_std_media = np.mean(desviaciones_std, axis=0)
    
    # Filtrar las características en un bucle que ajusta el umbral en caso de no obtener características
    while True:
        # Obtener índices booleanos de características con desviación estándar menor al umbral
        indices_utiles = desviaciones_std_media < umbral
        
        # Aplicar el filtro a la base de datos de voz
        base_datos_voz_filtrado = base_datos_voz[:, indices_utiles]
        
        # Comprobar si se obtuvieron características filtradas
        if base_datos_voz_filtrado.shape[1] > 0:
            break  # Salir del bucle si hay al menos una característica seleccionada
        else:
            umbral += 0.05  # Aumentar el umbral en 0.05 si no hay características útiles
    
    return base_datos_voz_filtrado

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
    knn = KNeighborsClassifier(n_neighbors=7)
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
    # Ajustar n_components basado en el número de características después del filtrado
    n_componentes = min(2, base_datos_voz.shape[1])
    pca = PCA(n_components=n_componentes)  # Reducir a 2D o 1D si solo hay una característica
    
    reduccion = pca.fit_transform(base_datos_voz)

    # Convertir etiquetas a números (por ejemplo, 'papa' -> 0, 'zanahoria' -> 1)
    etiquetas_unicas = list(set(etiquetas_voz))  # Obtener etiquetas únicas
    etiquetas_numericas = [etiquetas_unicas.index(label) for label in etiquetas_voz]

    # Crear un gráfico con cada verdura en un color distinto
    plt.figure(figsize=(10, 7))
    colores = plt.cm.viridis(np.linspace(0, 1, len(etiquetas_unicas)))
    for i, etiqueta in enumerate(etiquetas_unicas):
        indices = [j for j, e in enumerate(etiquetas_voz) if e == etiqueta]
        plt.scatter(reduccion[indices, 0], reduccion[indices, 1], color=colores[i], label=etiqueta)

    # Configurar el gráfico
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.title('Visualización PCA de las características de voz')
    plt.legend(title="Clase de verdura")
    plt.show()


# Guardar características de audio en CSV y calcular medias
def guardar_caracteristicas_y_medias_en_csv():
    ruta_voz = 'base_datos_voz/'
    archivo_csv = 'caracteristicas_audio_con_medias.csv'
    
    # Diccionario para almacenar características por verdura
    caracteristicas_por_verdura = defaultdict(list)

    # Crear y escribir encabezados en el archivo CSV
    with open(archivo_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        encabezados = ['archivo'] + [f'mfcc_{i}' for i in range(8)] + [f'chroma_{i}' for i in range(12)] + [f'spectral_contrast_{i}' for i in range(7)]
        writer.writerow(encabezados)
        
        # Procesar cada archivo de audio y almacenar características por verdura
        for archivo in os.listdir(ruta_voz):
            if archivo.endswith('.wav'):
                verdura = archivo.split('_')[0]  # Etiqueta de verdura basada en el nombre del archivo
                caracteristicas = extraer_caracteristicas_completas(os.path.join(ruta_voz, archivo))
                caracteristicas_redondeadas = np.round(caracteristicas, 3)  # Redondear a 3 decimales
                writer.writerow([archivo] + list(caracteristicas_redondeadas))
                caracteristicas_por_verdura[verdura].append(caracteristicas)
    
    # Calcular medias de características por verdura y guardarlas
    with open(archivo_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for verdura, caracteristicas in caracteristicas_por_verdura.items():
            media_caracteristicas = np.mean(caracteristicas, axis=0)
            media_caracteristicas_redondeadas = np.round(media_caracteristicas, 3)  # Redondear a 3 decimales
            writer.writerow([f'{verdura}_media'] + list(media_caracteristicas_redondeadas))

    print(f"Características y medias de audio guardadas en {archivo_csv}")

# ------------------------------- PROGRAMA PRINCIPAL ------------------------------

if __name__ == "__main__":
    # Paso 1: Cargar las bases de datos
    print("Cargando bases de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz_completa()

    # Paso 2: Filtrar características por clase
    print("Filtrando características con baja desviación estándar en cada clase...")
    base_datos_voz_filtrado = filtrar_caracteristicas_por_clase(base_datos_voz, etiquetas_voz)

    # Paso 3: Entrenar el modelo Knn para reconocimiento de voz
    print("Entrenando Knn para reconocimiento de voz...")
    knn = entrenar_knn(base_datos_voz_filtrado, etiquetas_voz)

    # Visualizar los datos con PCA
    print("Visualizando características de voz con PCA...")
    visualizar_datos(base_datos_voz_filtrado, etiquetas_voz)

    # Paso 4: Capturar audio directamente desde el micrófono
    # archivo_voz_test = grabar_audio_desde_microfono()  # Graba el audio desde el micrófono

    # Paso 5: Reconocer la verdura con el modelo entrenado
    # verdura_reconocida = reconocer_voz(knn, archivo_voz_test)
    # print(f"Verdura reconocida por voz: {verdura_reconocida}")

    # Paso 6: Mostrar la imagen correspondiente a la palabra reconocida
    # mostrar_imagen(verdura_reconocida)

    # Guardar características y medias en el archivo CSV
    guardar_caracteristicas_y_medias_en_csv()
