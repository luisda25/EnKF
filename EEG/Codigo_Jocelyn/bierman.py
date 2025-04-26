#Código creado por Jocelyn Ileana Balderas Sánchez

import numpy as np
from scipy import linalg as lin

def bierman_filter(S, H, R, x, y):
    """
    Aplica el filtro UD de Bierman para actualizar la estimación de estado y covarianza.

    Parámetros:
        S (np.ndarray): Factor triangular inferior de la matriz de covarianza original.
        H (np.ndarray): Matriz de observación.
        R (np.ndarray): Matriz de covarianza del ruido de medición.
        x (np.ndarray): Vector de estado actual.
        y (np.ndarray): Vector de medición.

    Retorna:
        S_new (np.ndarray): Factor triangular inferior actualizado.
        x_new (np.ndarray): Vector de estado actualizado.
    """
    # Reconstruir la matriz de covarianza original
    P = S @ S.T
    
    # Descomposición LDL de la matriz de covarianza
    U, D, _ = lin.ldl(P, lower=False)
    n = len(x)
    
    # Configurar variables iniciales
    b, p = np.zeros((n, 1)), np.zeros((n, 1))
    H_temp = np.reshape(H, (1, n)) if H.ndim == 1 else np.reshape(H.diagonal(), (1, n))
    D_new, U_new = np.zeros((n, n)), np.zeros((n, n))
    
    # Calcular intermedios
    f = U.T @ H_temp.T  # n x 1
    a = R[0, 0]
    v = np.array([D[i, i] * f[i] for i in range(n)])
    
    # Actualizar D y U mediante Bierman
    for k in range(n):
        a_new = a + f[k] * v[k]
        D_new[k, k] = D[k, k] * (a / a_new)
        b[k] = v[k]
        p[k] = -f[k] / a
        
        for j in range(k):
            U_new[j, k] = U[j, k] + b[j] * p[k]
            b[j] += U[j, k] * v[k]
        
        a = a_new
    
    # Ganancia de Kalman
    K = b / a
    
    # Predicción de la medición y actualización del estado
    y_pred = np.sum(y) / np.count_nonzero(H_temp == 1)
    x_new = x + K.flatten() * (y_pred - H_temp @ x)
    
    # Actualización del factor triangular inferior
    S_new = U_new.T @ lin.fractional_matrix_power(D_new, 0.5)
    
    return S_new, x_new