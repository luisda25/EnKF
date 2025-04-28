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

def householder_transform(F, Q, S):
    """
    Aplica el método de Householder para transformar matrices.

    Parámetros:
        F (np.ndarray): Matriz de entrada F.
        Q (np.ndarray): Matriz de covarianza Q.
        S (np.ndarray): Matriz de entrada S.

    Retorna:
        np.ndarray: La parte superior izquierda de la matriz transformada.
    """
    # Transpuestas y transformaciones iniciales
    F_trans = F.T
    S_trans = S.T
    Q_trans = np.sqrt(Q).T
    
    # Construir la matriz concatenada
    At = np.concatenate((F_trans @ S_trans, Q_trans), axis=0)
    n = At.shape[1]  # Número de columnas
    
    # Inicializar vectores auxiliares
    u = np.zeros((2 * n, 1))
    y = np.zeros((n, 1))
    
    for k in range(n):
        # Condición inicial
        if At[k, k] == 0:
            continue
        
        # Selección de columna y escalar
        Akk = At[:, k]
        scalA = At[k, k]
        
        # Cálculo de sigma
        sigma = -np.sign(scalA) * np.sqrt(np.sum(At[k:, k] ** 2))
        
        # Cálculo de beta
        beta = 1 / (sigma * (sigma + scalA))
        
        # Construcción del vector u
        u[:k, 0] = 0
        u[k, 0] = sigma + scalA
        u[k+1:, 0] = At[k+1:, k]
        
        # Construcción del vector y
        y[:k, 0] = 0
        y[k, 0] = 1
        y[k+1:, 0] = beta * (u.T @ At[:, k+1:])
        
        # Actualizar la matriz At
        At -= u @ y.T
    
    # Retornar la parte superior izquierda de la matriz transformada
    return At[:n, :n]

def combined_filter(F, Q, S, H, R, x, y):
    """
    Combina el filtro de Bierman y la transformación de Householder para análisis de señales.

    Parámetros:
        F, Q, S: Matrices para el filtro de Householder.
        H, R, x, y: Variables para el filtro de Bierman.

    Retorna:
        tuple: Resultado del filtro de Householder y el filtro de Bierman.
    """
    # Paso 1: Transformación de Householder
    householder_result = householder_transform(F, Q, S)
    
    # Paso 2: Filtro de Bierman
    S_new, x_new = bierman_filter(S, H, R, x, y)
    
    return householder_result, S_new, x_new