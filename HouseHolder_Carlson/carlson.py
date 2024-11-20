import numpy as np
import math

def carlson_filter_restructured(prevS, H, R, x, y):
    """
    Reimplementación del método de Carlson para filtro de Kalman.
    """
    
    n = len(x)
    m = len(y)
    x_prev = x

    # Aseguramos que prevS sea triangular inferior
    if not np.allclose(prevS, np.tril(prevS)):
        prevS = np.tril(prevS)

    S_prev = prevS

    for j in range(m):
        S = np.zeros_like(S_prev)  # Inicializa S para esta iteración
        H_j = H if H.ndim == 1 else H[j]  # Selecciona la fila correspondiente
        phi = S_prev @ H_j.reshape(-1, 1)  # Proyección de H_j en el espacio de S_prev

        d_prev = np.var(R[j])  # Inicialización del escalar
        e_prev = np.zeros((n, 1))  # Inicialización del vector auxiliar

        # Actualización columna por columna
        for i in range(n):
            d_next = d_prev + phi[i, 0] ** 2
            b = math.sqrt(d_prev / d_next)
            c = phi[i, 0] / math.sqrt(d_prev * d_next)

            e_next = e_prev + S_prev[:, i].reshape(-1, 1) * phi[i, 0]
            S[:, i] = S_prev[:, i] * b - e_prev.flatten() * c

            # Actualizamos para la siguiente iteración
            d_prev = d_next
            e_prev = e_next

        # Actualización del estado estimado
        innovation = y[j] - H_j @ x_prev
        x_next = x_prev + e_prev.flatten() * (innovation / d_prev)

        # Preparar para la próxima iteración
        S_prev = S
        x_prev = x_next

    return S, x_next


# Ejemplo de uso
prevS = np.array([[1, 0], [2, 1]])
H = np.array([[1, 0], [0, 1]])
R = np.array([[0.5, 0], [0, 0.2]])
x = np.array([0.5, 1.0])
y = np.array([1.2, 0.8])

S, x_next = carlson_filter_restructured(prevS, H, R, x, y)
print("Nueva matriz triangular inferior (S):")
print(S)
print("Nuevo estado estimado (x_next):")
print(x_next)