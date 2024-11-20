import numpy as np

def householder(A):
    """
    Calcula la raíz cuadrada de una matriz simétrica positiva definida A
    usando transformaciones de Householder.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)
    
    for k in range(n - 1):
        # Vector para reflejar
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        v = x.copy()
        v[0] += np.sign(x[0]) * norm_x
        v /= np.linalg.norm(v)
        
        # Matriz de Householder
        H = np.eye(n)
        H[k:, k:] -= 2.0 * np.outer(v, v)
        
        # Aplicar transformación
        R = H @ R
        Q = Q @ H.T

    # R ahora es triangular superior. La raíz cuadrada es su transpuesta inferior.
    return np.tril(R.T)

# Ejemplo: matriz simétrica positiva definida
A = np.array([[4, 2], [2, 3]])
L = householder(A)
print(L)