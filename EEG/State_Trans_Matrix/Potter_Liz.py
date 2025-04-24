import numpy as np
import math

def Potter_Algorithm(S, H, R, x, y):
    lenX = len(x)
    lenH = len(H)
    I = np.eye(lenX)
    
    S_i = S.copy()
    x_i = x.copy()

    for i in range(lenH):        
        H_i = H[i].reshape(1, lenX)  # Suponiendo H es (lenH, lenX)
        y_i = y[i].item()            # Suponiendo y es (lenH, 1)
        R_i = R[i] if np.isscalar(R[i]) else np.var(R[i])  # Usa varianza si no es escalar

        phi = np.dot(S_i.T, H_i.T)
        a = 1 / (np.dot(phi.T, phi) + R_i)
        gammaPlus = a / (1 + math.sqrt(a * R_i))

        Splus = np.dot(S_i, (I - gammaPlus * np.dot(phi, phi.T)))
        k = np.dot(S_i, phi)
        xPlus = x_i + k.dot(a * (y_i - H_i.dot(x_i)))

        S_i = Splus
        x_i = xPlus
   
    return S_i, x_i
