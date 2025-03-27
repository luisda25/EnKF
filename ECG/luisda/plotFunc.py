from scipy.io import loadmat
import numpy as np
import heartpy as hp

def plot(data_path, plot_path, n_title):
    data = loadmat(data_path)
    ecg_data = data["Cn"][0][0].flatten()
    cn = data["Cn"]
    #ecg_data = np.hstack([np.array(x).flatten() for x in cn.ravel()]).astype(np.float64)
    rate = 256.0
    width = 15
    overlap = 0.5

    w, m = hp.process(ecg_data, rate)
    w, m = hp.process_segmentwise(ecg_data, sample_rate=rate, segment_width=width, segment_overlap=overlap, calc_freq=True, reject_segmentwise=False,high_precision=True)

    
    hp.plotter(w, m, title="Prueba")
    
    plt.savefig('plot_1.jpg')

    plt.show()

