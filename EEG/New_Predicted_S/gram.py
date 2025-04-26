import numpy as np

def modified_gram_schmidt(F, Q, S):
    """
    Implementa el método de Gram-Schmidt Modificado usando una matriz augmentada,
    para construir una matriz triangular superior W.
    """

    F_T = F.T
    S_T = S  

    # Raíz cuadrada de Q y luego transponer
    Q_T = np.sqrt(Q).T

    # Multiplicar F_T y S_T (producto matricial)
    first_product = F_T @ S_T

    # Construir matriz augmentada: apilar verticalmente el producto y la raíz cuadrada de Q
    augmented_matrix = np.vstack((first_product, Q_T))

    # Obtener número de columnas de la matriz augmentada
    num_cols = augmented_matrix.shape[1]

    # Inicializar matriz triangular superior W
    W = np.zeros((num_cols, num_cols))

    # Crear copia temporal de la matriz augmentada para actualizarla
    temp_augmented = np.zeros_like(augmented_matrix)

    # Algoritmo de Gram-Schmidt Modificado
    for k in range(num_cols):
        # Tomar la k-ésima columna de la matriz actual
        col_k = augmented_matrix[:, k]

        # Calcular norma (longitud) de la columna
        norm_k = np.sqrt(np.dot(col_k.T, col_k))

        for j in range(num_cols):
            if j == k:
                # Elementos diagonales de W son la norma
                W[k, j] = norm_k
            elif j < k:
                # Elementos por debajo de la diagonal son 0
                W[k, j] = 0
            else:
                # Elementos por encima de la diagonal son coeficientes de proyección
                W[k, j] = np.dot(col_k.T, augmented_matrix[:, j]) / norm_k

        # Ortogonalizar las siguientes columnas
        for j in range(k + 1, num_cols):
            temp_augmented[:, j] = augmented_matrix[:, j] - (W[k, j] * col_k) / norm_k

        # Actualizar la matriz augmentada para el siguiente paso
        augmented_matrix = temp_augmented

    return W
