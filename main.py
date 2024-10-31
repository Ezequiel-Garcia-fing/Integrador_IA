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
from mpl_toolkits.mplot3d import Axes3D  # Importar para visualizar gráficos en 3D
from scipy.stats import f_oneway
# ----------------------------- FUNCIONES AUXILIARES -----------------------------

# Nueva función para segmentar el audio
def segmentar_audio(archivo_audio, num_segmentos=10):
    audio_data, sample_rate = librosa.load(archivo_audio)
    audio_data, _ = librosa.effects.trim(audio_data)
    duracion_segmento = len(audio_data) // num_segmentos
    audio_data = librosa.util.fix_length(audio_data, size=num_segmentos * duracion_segmento)
    segmentos = [audio_data[i * duracion_segmento: (i + 1) * duracion_segmento] for i in range(num_segmentos)]
    return segmentos, sample_rate


# Modificación de extraer_caracteristicas_completas para usar solo el segmento 9 en ciertas palabras
# Modificación de extraer_caracteristicas_completas para usar segmentos 6, 7 y 9 en ciertas palabras
def extraer_caracteristicas_completas(archivo_audio, umbral_amplitud=0.030):
    palabra = os.path.basename(archivo_audio).split('_')[0]
    segmentos, sample_rate = segmentar_audio(archivo_audio)

    segmentos_filtrados = [
        seg for seg in segmentos if np.mean(np.abs(seg)) > umbral_amplitud 
    ]

    if palabra in ["choclo", "zanahoria"]:
        if len(segmentos) > 8:
            segmentos_filtrados = [segmentos[5], segmentos[6], segmentos[8]]

    if not segmentos_filtrados:
        print(f"Advertencia: archivo {archivo_audio} no tiene segmentos con suficiente amplitud.")
        return np.zeros(3 + 1 + 1)  # Tamaño ajustado sin Spectral Contrast

    mfccs_mean, zcr_mean, rms_mean = [], [], []
    for seg in segmentos_filtrados:
        mfccs = librosa.feature.mfcc(y=seg, sr=sample_rate, n_mfcc=5)
        mfccs_selected = mfccs[[0, 3, 4], :]  # Seleccionar los coeficientes 1, 4 y 5 (índices 0, 3 y 4)
        mfccs_mean.append(np.mean(mfccs_selected, axis=1))
        zcr = librosa.feature.zero_crossing_rate(y=seg)
        zcr_mean.append(np.mean(zcr))
        rms = librosa.feature.rms(y=seg)
        rms_mean.append(np.mean(rms))

    mfccs_mean = np.mean(mfccs_mean, axis=0)
    zcr_mean = np.mean(zcr_mean)
    rms_mean = np.mean(rms_mean)

    caracteristicas = np.hstack([mfccs_mean, zcr_mean, rms_mean])

    return caracteristicas

def encontrar_segmentos_caracteristicos(ruta_voz, palabras, num_segmentos=10):
    varianza_por_segmento = []

    for segmento in range(num_segmentos):
        caracteristicas_por_clase = []

        for palabra in palabras:
            caracteristicas_segmento = []
            for archivo in os.listdir(ruta_voz):
                if archivo.startswith(palabra) and archivo.endswith('.wav'):
                    segmentos, sample_rate = segmentar_audio(os.path.join(ruta_voz, archivo), num_segmentos)
                    mfccs = librosa.feature.mfcc(y=segmentos[segmento], sr=sample_rate, n_mfcc=5)
                    mfccs_selected = mfccs[[0, 3, 4], :]  # Coeficientes 1, 4 y 5
                    caracteristicas_segmento.append(np.mean(mfccs_selected, axis=1))
            caracteristicas_por_clase.append(np.mean(caracteristicas_segmento, axis=0))

        varianza_segmento = np.var(np.array(caracteristicas_por_clase), axis=0).mean()
        varianza_por_segmento.append((f"Segmento_{segmento + 1}", varianza_segmento))

    # Ordenar segmentos por varianza media entre clases (descendente)
    varianza_por_segmento.sort(key=lambda x: x[1], reverse=True)
    return varianza_por_segmento

def identificar_segmentos_caracteristicos_por_caracteristica(ruta_voz, palabras, num_segmentos=10):
    # Estructura para almacenar las características de cada segmento y palabra
    caracteristicas_por_segmento = {
        palabra: {f"Segmento_{i+1}": {'MFCC1': [], 'MFCC4': [], 'MFCC5': [], 'ZCR': [], 'RMS': []}
                  for i in range(num_segmentos)} for palabra in palabras
    }

    # Extraer características y organizarlas por palabra y segmento
    for palabra in palabras:
        for archivo in os.listdir(ruta_voz):
            if archivo.startswith(palabra) and archivo.endswith('.wav'):
                segmentos, sample_rate = segmentar_audio(os.path.join(ruta_voz, archivo), num_segmentos)

                for i, segmento in enumerate(segmentos):
                    mfccs = librosa.feature.mfcc(y=segmento, sr=sample_rate, n_mfcc=5)
                    zcr = librosa.feature.zero_crossing_rate(y=segmento)
                    rms = librosa.feature.rms(y=segmento)

                    # Guardar características en el diccionario por segmento y palabra
                    caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"]['MFCC1'].append(np.mean(mfccs[0, :]))
                    caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"]['MFCC4'].append(np.mean(mfccs[3, :]))
                    caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"]['MFCC5'].append(np.mean(mfccs[4, :]))
                    caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"]['ZCR'].append(np.mean(zcr))
                    caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"]['RMS'].append(np.mean(rms))

    # Calcular y mostrar la varianza media de cada característica por segmento
    print("Identificando los segmentos más característicos por característica para cada palabra...")
    for i in range(num_segmentos):
        varianza_media_segmento = {caracteristica: [] for caracteristica in ['MFCC1', 'MFCC4', 'MFCC5', 'ZCR', 'RMS']}
        for palabra in palabras:
            for caracteristica, valores in caracteristicas_por_segmento[palabra][f"Segmento_{i+1}"].items():
                varianza_media_segmento[caracteristica].append(np.var(valores))

        # Calcular y mostrar la varianza media de cada característica en el segmento
        print(f"\nSegmento_{i+1}:")
        for caracteristica, varianzas in varianza_media_segmento.items():
            varianza_media = np.mean(varianzas)
            print(f"{caracteristica}: varianza media = {varianza_media:.4f}")

# Modificación de graficar_onda_segmentada_promedio para usar segmentar_audio
def graficar_onda_segmentada_promedio(ruta_voz, palabras, num_segmentos=10):
    # Diccionario para almacenar las ondas promedio de cada segmento por palabra
    segmentos_promedio = {palabra: np.zeros((num_segmentos, 1)) for palabra in palabras}
    
    # Determinar la longitud mínima de los segmentos entre todos los archivos de cada palabra
    min_segment_length = None
    for palabra in palabras:
        for archivo in os.listdir(ruta_voz):
            if archivo.startswith(palabra) and archivo.endswith('.wav'):
                segmentos, _ = segmentar_audio(os.path.join(ruta_voz, archivo), num_segmentos)
                duracion_segmento = len(segmentos[0])
                if min_segment_length is None or duracion_segmento < min_segment_length:
                    min_segment_length = duracion_segmento

    for palabra in palabras:
        sum_segmentos = np.zeros((num_segmentos, min_segment_length))
        num_audios = 0

        for archivo in os.listdir(ruta_voz):
            if archivo.startswith(palabra) and archivo.endswith('.wav'):
                segmentos, _ = segmentar_audio(os.path.join(ruta_voz, archivo), num_segmentos)

                # Ajustar cada segmento a la longitud mínima
                for i in range(num_segmentos):
                    segmento = segmentos[i]
                    if len(segmento) > min_segment_length:
                        segmento = segmento[:min_segment_length]
                    elif len(segmento) < min_segment_length:
                        segmento = np.pad(segmento, (0, min_segment_length - len(segmento)), mode='constant')
                    
                    sum_segmentos[i] += segmento
                
                num_audios += 1

        # Calcular el promedio de cada segmento para la palabra actual
        if num_audios > 0:
            segmentos_promedio[palabra] = sum_segmentos / num_audios

    # Graficar los segmentos promedio para cada palabra
    plt.figure(figsize=(12, 10))
    for i, (palabra, segmentos) in enumerate(segmentos_promedio.items()):
        for j in range(num_segmentos):
            plt.subplot(len(palabras), num_segmentos, i * num_segmentos + j + 1)
            plt.plot(segmentos[j], color='blue')
            if j == 0:
                plt.ylabel(f"'{palabra}'")
            if i == 0:
                plt.title(f"Segmento {j + 1}")
            plt.xlabel("Tiempo")
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.show()




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

def seleccionar_caracteristicas_por_fscore(base_datos_voz, etiquetas_voz, top_n=5):
    # Obtener la lista de clases únicas en etiquetas_voz
    clases = np.unique(etiquetas_voz)
    
    # Crear un diccionario para almacenar las características por clase
    caracteristicas_por_clase = {clase: base_datos_voz[etiquetas_voz == clase] for clase in clases}
    
    # Calcular el F-score para cada característica
    f_scores = []
    for i in range(base_datos_voz.shape[1]):
        # Crear una lista con las características de cada clase en la posición 'i'
        datos_por_clase = [caracteristicas_por_clase[clase][:, i] for clase in clases]
        
        # Calcular el F-score usando ANOVA de una vía (f_oneway)
        f_score, _ = f_oneway(*datos_por_clase)
        f_scores.append(f_score)
    
    # Ordenar índices de características según el F-score, en orden descendente
    indices_top = np.argsort(f_scores)[-top_n:][::-1]
    f_scores_top = [f_scores[i] for i in indices_top]
    
    print("Características seleccionadas con F-scores:")
    for idx, f_score in zip(indices_top, f_scores_top):
        print(f"Característica {idx} - F-score: {f_score}")
    
    # Seleccionar solo las características con los F-scores más altos
    base_datos_voz_filtrado = base_datos_voz[:, indices_top]
    
    return base_datos_voz_filtrado, indices_top

def calcular_y_guardar_promedios_caracteristicas(ruta_voz, palabras, num_segmentos=10, archivo_csv='promedios_caracteristicas.csv'):
    promedios_por_clase = {palabra: {f"Segmento_{i+1}": {'MFCC': [], 'ZCR': [], 'RMS': []} for i in range(num_segmentos)} for palabra in palabras}

    for palabra in palabras:
        for archivo in os.listdir(ruta_voz):
            if archivo.startswith(palabra) and archivo.endswith('.wav'):
                segmentos, sample_rate = segmentar_audio(os.path.join(ruta_voz, archivo), num_segmentos)
                
                for i, segmento in enumerate(segmentos):
                    mfccs = librosa.feature.mfcc(y=segmento, sr=sample_rate, n_mfcc=5)
                    mfccs_selected = mfccs[[0, 3, 4], :]  # Seleccionar los coeficientes 1, 4 y 5
                    zcr = librosa.feature.zero_crossing_rate(y=segmento)
                    rms = librosa.feature.rms(y=segmento)
                    
                    promedios_por_clase[palabra][f"Segmento_{i+1}"]['MFCC'].append(np.mean(mfccs_selected, axis=1))
                    promedios_por_clase[palabra][f"Segmento_{i+1}"]['ZCR'].append(np.mean(zcr))
                    promedios_por_clase[palabra][f"Segmento_{i+1}"]['RMS'].append(np.mean(rms))

    with open(archivo_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Clase', 'Segmento', 'MFCC1', 'MFCC4', 'MFCC5', 'ZCR', 'RMS'])
        
        for palabra, segmentos in promedios_por_clase.items():
            for segmento, caracteristicas in segmentos.items():
                mfccs_mean = np.mean(caracteristicas['MFCC'], axis=0)
                zcr_mean = np.mean(caracteristicas['ZCR'])
                rms_mean = np.mean(caracteristicas['RMS'])
                
                writer.writerow([
                    palabra, segmento, *mfccs_mean, zcr_mean, rms_mean
                ])

    print(f"Archivo CSV guardado como '{archivo_csv}'")



# ------------------------------- PROGRAMA PRINCIPAL ------------------------------

if __name__ == "__main__":
    # Paso 1: Cargar las bases de datos de voz
    print("Cargando bases de datos de voz...")
    base_datos_voz, etiquetas_voz = cargar_base_datos_voz_completa()

    base_datos_voz, etiquetas_voz = cargar_base_datos_voz_completa()
    base_datos_voz_filtrado, indices_top = seleccionar_caracteristicas_por_fscore(base_datos_voz, etiquetas_voz, top_n=5)
    # Paso 2: Filtrar características por clase
    print("Filtrando características con baja desviación estándar en cada clase...")
    base_datos_voz_filtrado = filtrar_caracteristicas_por_clase(base_datos_voz, etiquetas_voz)

    # Paso 3: Entrenar el modelo Knn para reconocimiento de voz
    print("Entrenando Knn para reconocimiento de voz...")
    knn = entrenar_knn(base_datos_voz_filtrado, etiquetas_voz)

    # Visualizar los datos con PCA
    print("Visualizando características de voz con PCA...")
    visualizar_datos(base_datos_voz_filtrado, etiquetas_voz)

    # Generar archivo CSV con los promedios de las características
    ruta_voz = 'base_datos_voz/'
    palabras = ['choclo', 'berenjena', 'zanahoria', 'papa']
    print("Calculando y guardando promedios de características en CSV...")
    calcular_y_guardar_promedios_caracteristicas(ruta_voz, palabras, num_segmentos=10, archivo_csv='promedios_caracteristicas.csv')

    # Graficar los segmentos promedio de cada clase
    print("Graficando segmentos promedio por clase...")
    graficar_onda_segmentada_promedio(ruta_voz, palabras, num_segmentos=10)

    # Encontrar los segmentos más característicos de cada palabra
    print("Identificando los segmentos más característicos para cada palabra...")
    segmentos_caracteristicos = encontrar_segmentos_caracteristicos(ruta_voz, palabras, num_segmentos=10)
    for segmento, varianza in segmentos_caracteristicos:
        print(f"{segmento}: varianza media = {varianza}")

    identificar_segmentos_caracteristicos_por_caracteristica(ruta_voz, palabras, num_segmentos=10)
