from scipy.io import loadmat
import heartpy as hp 
from plotFunc import plot 

# Procesamiento de datos usando segmentwise de los 10 archivos
main_path = "/home/luisda/Documents/test/dataset/"
file = "subject01.mat"
path = main_path + file
title = "Número plot_1"
plot(path, "./plot_1/", title)
