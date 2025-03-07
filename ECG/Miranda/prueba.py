import heartpy as hp # Heartpy

import matplotlib.pyplot as plt # Para generar gráficos
from scipy.io import loadmat # Para archivo de MATLAB
import numpy as np # Convertir datos

from scipy.io import loadmat
import numpy as np

# Cargar el archivo y convertirlo
file_path = "subject03.mat" 
mat_data = loadmat(file_path) # Cargar el archivo .mat
Cn = mat_data["Cn"] # Extraer la matriz Cn
hrdata = np.hstack([np.array(x).flatten() for x in Cn.ravel()]).astype(np.float64)

sample_rate = 256.0 # This set is sampled at 256Hz

# Returns two dictionaries: working data and measures
working_data, measures = hp.process(hrdata, sample_rate) 

# Plotter
hp.plotter(working_data, measures, title = 'Ejemplo')

# Save the graph with Matplotlib
plt.savefig('plot_3.jpg')

# Print values
print(f"BPM: {measures['bpm']}") # returns BPM value
print(f"HRV Measure: {measures['rmssd']}") # returns RMSSD HRV measure

# Show graph
plt.show()
