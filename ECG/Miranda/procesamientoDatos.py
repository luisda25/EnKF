import heartpy as hp # Heartpy

import matplotlib.pyplot as plt # Para generar gráficos
from scipy.io import loadmat # Para archivo de MATLAB
import numpy as np # Convertir datos

from scipy.io import loadmat
import numpy as np

# Cargar el archivo
subject = "subject03.mat"
file_path = "/Users/mirandaurbansolano/Documents/GitHub/" + subject

# Convertir el archivo
mat_data = loadmat(file_path) # Cargar el archivo .mat
Cn = mat_data["Cn"] # Extraer la matriz Cn
hrdata = np.hstack([np.array(x).flatten() for x in Cn.ravel()]).astype(np.float64)

rate = 256.0  # La muestra esta sampleada a 256Hz

# Regresa dos diccionarios: working data y measures
working_data, measures = hp.process(hrdata, rate) 

working_data, measures = hp.process_segmentwise(
    hrdata,  
    sample_rate = rate,  
    segment_width = 15,  # 15 segundos  
    segment_overlap=0.5,  
    calc_freq=True,
    reject_segmentwise=False,  # No eliminar segmentos
    high_precision=True  # Más precisión  
)

print(f"Total peaks detected: {len(working_data['peaklist'])}")
print(f"Peaks eliminados: {len(working_data['removed_beats'])}")

# Plotter
#hp.plotter(working_data, measures, title = 'Ejemplo')

# Save the graph with Matplotlib
#plt.savefig('plot_3.jpg')

# Print values
#print(f"BPM: {measures['bpm']}") # returns BPM value
#print(f"HRV Measure: {measures['rmssd']}") # returns RMSSD HRV measure

# Show graph
#plt.show()
