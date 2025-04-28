#Código creado por Jocelyn Ileana Balderas Sánchez

import numpy as np

def householder_filter(F, Q, S):
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

def summatory(iteration, size, matrix):
    """
    Calcula la suma de los cuadrados de los elementos de una columna a partir de una fila dada.
    
    Parámetros:
        iteration (int): Índice de inicio.
        size (int): Tamaño del bloque de la matriz.
        matrix (np.ndarray): Matriz de entrada.
    
    Retorna:
        float: Suma de los cuadrados.
    """
    return np.sum(matrix[iteration:2 * size, iteration] ** 2)