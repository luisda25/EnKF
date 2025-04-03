import heartpy as hp # Heartpy

from scipy.io import loadmat # Para archivo de MATLAB
import numpy as np # Convertir datos
from scipy.interpolate import interp1d # Pra intrapolar los datos

# Cargar el archivo y convertirlo
def load_file(file_path):
    data = loadmat(file_path) # Cargar el archivo .mat
    Cn = data["Cn"] # Extraer la matriz Cn
    hrdata = data["Cn"][0][0].flatten()
    return hrdata
    
# Analizar la señal para obtener datos
def analyze_data(subject, file_path):
    hrdata = load_file(file_path)

    # Declaración de variables para el análisis
    sample_rate = 256.0 # Los datos están sampleados a 256Hz
    width = 15 # Ancho de cada segmento
    overlap = 0.5 # 

    # Interpolar datos
    #time_original = np. linspace(0, len(hrdata)/sample_rate, len (hrdata)) # Crear un eje de tiempo original
    #new_time = np. linspace(0, len(hrdata)/sample_rate, len(hrdata) * 2) # Crear un nuevo eje de tiempo con mayor resolución
    #interpolator = interp1d(time_original, hrdata, kind='cubic')
    #new_hrdata = interpolator(new_time)

    # Procesar la señal
    working_data, measures = hp.process(hrdata, sample_rate) 

    working_data, measures = hp.process_segmentwise(
        hrdata,  
        sample_rate = sample_rate,  
        segment_width = width,  # 15 segundos  
        segment_overlap = overlap,  
        calc_freq = True,
        reject_segmentwise = True,  # No eliminar segmentos
        high_precision = True)

    # Calcular los promedios de BPM y HRV
    bpm_avg = np.mean(measures['bpm'])
    hrv_avg = np.mean(measures['rmssd'])
    lf_hf_avg = np.mean(measures['lf/hf'])

    # Imprimir los valores obtenidos
    print(f"Información de " + subject)
    print(f"Promedio de BPM: {bpm_avg:.2f}")
    print(f"Promedio de HRV (RMSSD): {hrv_avg:.2f}")
    print(f"Promedio de lf/hf: {lf_hf_avg:.2f}")

     # Imprimir los valores obtenidos
    tit = "Datos de " + subject
    bpm_text = "Promedio de BPM: " + str(round(bpm_avg, 2)) + "\n"
    hrv_text = "Promedio de HRV (RMSSD): " + str(round(hrv_avg, 2)) + "\n"
    lf_hf_text = "Promedio de lf/hf: " + str(round(lf_hf_avg, 2)) + "\n"

    # Escribir en el archivo
    resultados_sujeto = "resultados_" + subject + ".txt"
    with open(resultados_sujeto, "a") as file:
        file.write(tit + "\n")
        file.write(bpm_text)
        file.write(hrv_text)
        file.write(lf_hf_text)
        file.write("")


subject = "subject04"
subject_file = "/Users/mirandaurbansolano/Documents/GitHub/" + subject + ".mat"

analyze_data(subject, subject_file)