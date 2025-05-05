import math

# Matriz de covarianza inicial
P = [
    [4, 1, 0, 0, 0],
    [1, 3, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 1, 2, 1],
    [0, 0, 0, 1, 1]
]


n = len(P)
D = [P[i][i] for i in range(n)]  # Diagonal de P
U = [[0 if i != j else 1 for j in range(n)] for i in range(n)]  # Matriz identidad para U

#metodo de Bierman
for k in range(n):
    # Calculo de los factores para las rotaciones de Givens
    for i in range(k + 1, n):
        if P[k][k] != 0:
            c = P[k][k] / math.sqrt(P[k][k] ** 2 + P[i][k] ** 2)
            s = P[i][k] / math.sqrt(P[k][k] ** 2 + P[i][k] ** 2)
