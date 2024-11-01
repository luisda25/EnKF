import numpy as np

def householder(A):
    """
    Aplica la transformación de Householder a la matriz A para calcular su raíz cuadrada.
    Devuelve una matriz triangular superior R tal que R * R^T ≈ A.
    """
    m, n = A.shape
    R = A.astype(float)

    for i in range(min(m, n)):
        # Crear el vector de Householder para la columna i
        x = R[i:, i]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        u = x - e1
        
        norm_u = np.linalg.norm(u)
        if norm_u != 0:
            u = u / norm_u  
        
        # Aplicar la transformación de Householder
        R[i:, :] -= 2 * np.outer(u, u @ R[i:, :])
    
    return np.triu(R)  


A = np.array([
    [4, 1, 2],
    [1, 2, 3],
    [2, 3, 6]
])

S = householder(A)
print(A)
print(S)
