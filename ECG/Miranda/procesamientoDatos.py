import heartpy as hp # Heartpy

from scipy.io import loadmat # Para archivo de MATLAB
import numpy as np # Convertir datos
from scipy.interpolate import interp1d # Pra intrapolar los datos

# Elección del sujeto
def choose_subject(num, interpolados):
    subject = "subject" + num
    subject_file = "./DatosSujetos/" + subject + ".mat"
    
    if interpolados == "y":
        interpolados = True
    elif interpolados == "n":
        interpolados = False
    else:
        return "Error"

    analyze_data(subject, subject_file, interpolados)

# Cargar el archivo y convertirlo
def load_file(file_path):
    data = loadmat(file_path)
    ecg_data = data["Cn"][0][0]  # Acceder a ECG (considerat que está en la 1era columna)
    hrdata = ecg_data.flatten()
    return hrdata
    
# Analizar la señal para obtener datos
def analyze_data(subject, file_path, interpolados):
    hrdata = load_file(file_path)

    # Declaración de variables para el análisis
    sample_rate = 256.0 # Los datos están sampleados a 256Hz
    width = 15 # Ancho de cada segmento
    overlap = 0.5 # 

    # Interpolar datos
    if interpolados == True:
        hrdata = hrdata.flatten()

        time_original = np.linspace(0, len(hrdata)/sample_rate, len(hrdata))
        new_time = np.linspace(0, len(hrdata)/sample_rate, len(hrdata) * 2)

        try:
            interpolator = interp1d(time_original, hrdata, kind='cubic')
            hrdata = interpolator(new_time)
            sample_rate *= 2 
            res = "resultadosInterpolados_"
        except Exception as e:
            print("Error al interpolar:", e)
    else: 
        res = "resultados_"

    # Procesar la señal
    working_data, measures = hp.process(hrdata, sample_rate) 

    working_data, measures = hp.process_segmentwise(
        hrdata,  
        sample_rate = sample_rate,  
        segment_width = width,  # 15 segundos  
        segment_overlap = overlap,  
        calc_freq = True,
        reject_segmentwise = True,  # No eliminar segmentos
        high_precision = True # Mayor precisión
    )

    # Imprimir los valores obtenidos
    tit = "Datos de " + subject + "\n"
    bpm_text = "Datos de BPM: " + str(measures['bpm']) + "\n"
    hrv_text = "Datos de HRV (RMSSD): " + str(measures['rmssd']) + "\n"
    lf_hf_text = "Datos de lf/hf: " + str(measures['lf/hf']) + "\n"

    # Escribir en el archivo
    resultados_sujeto = "Resultados/" + res + subject + ".txt"
    with open(resultados_sujeto, "w") as file:
        file.write(tit + "\n")
        file.write(bpm_text)
        file.write(hrv_text)
        file.write(lf_hf_text)
        file.write("")

    # Mensaje final
    print(f"\n\nInformación de " + subject + " analizada ")

num = input("Escoge un sujeto con formato '0X': ")
interpolados = input("¿Se deben interpolar? [y/n]: ")
choose_subject(num, interpolados)