import math
import numpy as np
from scipy import linalg as lin

def bierman_filter(P, H, R, x, y):
    # Descomponer P en U y D usando descomposición LDL
    U, D, perm = lin.ldl(P, lower=False)
    
    # Inicializar matrices y variables necesarias
    D_new = np.zeros_like(D)
    U_new = np.zeros_like(U)
    b, p = np.zeros(len(x)), np.zeros(len(x))
    H_temp = H if H.shape[0] == len(x) else np.diag(H).reshape(1, -1)

    # Cálculo del vector intermedio f
    f = np.dot(U.T, H_temp.T).flatten()  # Vector de ganancia intermedia

    # Inicializar variables para la iteración
    a = R[0, 0]
    v = np.array([D[i, i] * f[i] for i in range(len(x))])

    # Iteración de actualización de D y U
    for k in range(len(x)):
        # Actualizar 'a' y 'D'
        a_new = a + f[k] * v[k]
        D_new[k, k] = D[k, k] * (a / a_new)
        
        # Calcular b_k y p_k
        b[k] = v[k]
        p[k] = -f[k] / a
        
        # Actualizar U y b
        for j in range(k):
            U_new[j, k] = U[j, k] + b[j] * p[k]
            b[j] = b[j] + U[j, k] * v[k]
        
        # Preparar para la siguiente iteración
        a = a_new
    
    # Calcular la ganancia de Kalman K
    K = b / a
    
    # Estimar el estado actualizado x_new
    y_pred = np.dot(H_temp, x).flatten()
    x_new = x + K * (y - y_pred)
    
    # Nueva covarianza S usando U y D actualizados
    S_new = U_new.T @ np.diag(np.sqrt(np.diag(D_new)))
    
    return S_new, x_new

# Matriz de covarianza inicial
P = np.array([
    [4, 1, 0, 0, 0],
    [1, 3, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 1, 2, 1],
    [0, 0, 0, 1, 1]
])

# Otros parámetros de ejemplo
H = np.eye(5)  # Matriz de observación (identidad)
R = np.array([[0.5]])  # Ruido de medición
x = np.array([1, 1, 1, 1, 1])  # Estado estimado inicial
y = np.array([2, 2, 2, 2, 2])  # Observación

# Ejecutar el filtro de Bierman
S_new, x_new = bierman_filter(P, H, R, x, y)

print("Nueva matriz de covarianza (S):\n", S_new)
print("Nuevo estado estimado (x):\n", x_new)
