import heartpy as hp # Heartpy

import matplotlib.pyplot as plt # Para generar gráficos
from scipy.io import loadmat # Para archivo de MATLAB
import numpy as np # Convertir datos

from scipy.io import loadmat
import numpy as np

# Cargar el archivo y convertirlo
subject = "subject03.mat"
file_path = "/Users/mirandaurbansolano/Documents/GitHub/" + subject
mat_data = loadmat(file_path) # Cargar el archivo .mat
Cn = mat_data["Cn"] # Extraer la matriz Cn
hrdata = np.hstack([np.array(x).flatten() for x in Cn.ravel()]).astype(np.float64)

sample_rate = 256.0 # This set is sampled at 256Hz
filter_type = 'lowpass'

# Filtrar datos para low y high pass
filtered = hp.filter_signal(hrdata, cutoff = 12, sample_rate = sample_rate, order = 4, filtertype=filter_type)

# Returns two dictionaries: working data and measures
working_data, measures = hp.process(filtered, sample_rate) 

# Plotter
hp.plotter(working_data, measures, title = 'Ejemplo')

# Save the graph with Matplotlib
nombre_plot = 'plotPrueba_3_' +  filter_type + '.jpg'
plt.savefig(nombre_plot)

# Print values
print(f"BPM: {measures['bpm']}") # returns BPM value
print(f"HRV Measure: {measures['rmssd']}") # returns RMSSD HRV measure

# Show graph
plt.show()