import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.interpolate import interp1d

def graficar_rmssd(nombre_archivo, num, width=15, overlap=0.5):
    # Abrir archivo con información del sujeto (ya analizado)
    with open(nombre_archivo, 'r') as f:
        lines = f.readlines()

    # Buscar línea con los valores de RMSSD
    rmssd_line = next((line for line in lines if 'Datos de HRV (RMSSD)' in line), None)
    if rmssd_line is None:
        print("No se encontró HRV (RMSSD) en el archivo.")
        return

    try:
        # Extraer números como flotantes
        numeros = re.findall(r"[-+]?\d*\.\d+|\d+", rmssd_line)
        rmssd_vals = np.array([float(num) for num in numeros])
    except Exception as e:
        print("Error al procesar los valores de HRV (RMSSD):", e)
        return

    # Crear eje de tiempo
    step = width * (1 - overlap)
    times = np.arange(0, len(rmssd_vals) * step, step)

    # Interpolación
    try:
        interpolador = interp1d(times, rmssd_vals, kind='cubic', fill_value='extrapolate')
        tiempo_interp = np.linspace(times[0], times[-1], 500)
        rmssd_interp = interpolador(tiempo_interp)
    except Exception as e:
        print("Error al interpolar RMSSD:", e)
        return

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(times, rmssd_vals, 'o', label='RMSSD original', color='darkblue')
    plt.plot(tiempo_interp, rmssd_interp, '-', label='Interpolación', color='orange')
    plt.title(f'HRV (RMSSD) vs Tiempo\nde sujeto {num}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('RMSSD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
num = input("Escoge un sujeto con formato '0X': ")
archivo = "Resultados/resultadosInterpolados_subject" + num + ".txt"
graficar_rmssd(archivo, num)