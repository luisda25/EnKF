import numpy as np

def gram_schmidt_modified(A, normalize=True):
    """
    Realiza el método de Gram-Schmidt Modificado para ortogonalizar los vectores columna de A.
    
    Parámetros:
    A (numpy.ndarray): Matriz de entrada con vectores columna que se quieren ortogonalizar.
    normalize (bool): Si es True, normaliza los vectores resultantes para obtener una base ortonormal.
    
    Retorna:
    numpy.ndarray: Matriz con vectores ortogonales (o ortonormales si normalize=True).
    """
    # Dimensiones de la matriz A
    rows, cols = A.shape
    
    # Inicializa una matriz para almacenar los vectores ortogonales
    U = np.zeros((rows, cols))
    
    for i in range(cols):
        # Empieza con el vector original v_i
        u_i = A[:, i]
        
        # Resta las proyecciones de u_i en los vectores ortogonales anteriores
        for j in range(i):
            u_i = u_i - np.dot(U[:, j], A[:, i]) / np.dot(U[:, j], U[:, j]) * U[:, j]
        
        # Asigna el vector ortogonalizado a la columna correspondiente en U
        U[:, i] = u_i
    
    # Si se desea una base ortonormal, normaliza cada vector
    if normalize:
        for i in range(cols):
            U[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    
    return U

# Matriz de ejemplo con una sola columna
A = np.array([[1],
              [2],
              [3]], dtype=float)

# Ejecutar el método de Gram-Schmidt
result = gram_schmidt_modified(A)

print("Matriz ortogonalizada (o base ortonormal si está normalizada):")
print(result)