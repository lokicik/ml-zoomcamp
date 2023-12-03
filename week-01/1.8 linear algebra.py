import numpy as np
# Vector operations
u = np.array([2, 4, 5, 6])
2 * u
v = np.array([1, 0, 0, 2])
u + v

# Vector multiplication
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    result = 0.0
    n = u.shape[0]
    for i in range(n):
        result = result + u[i] * v[i]
    return result

vector_vector_multiplication(u, v)

u.dot(v)

# Matrix-vector multiplication
U = np.array([
    [2,4,5,6],
    [1,2,1,2],
    [3,1,2,1]
])
U
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]

    num_rows = U.shape[0]
    result = np.zeros(num_rows)
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    return result

matrix_vector_multiplication(U,v)
U.dot(v)

# Matrix-matrix multiplication
V = np.array([
    [1,1,2],
    [0,0.5,1],
    [0,2,1],
    [2,1,0]
])
def matrix_matrix_multiplicaton(U, V):
    assert U.shape[1] == V.shape[0]

    num_rows = U.shape[0]
    num_cols = V.shape[1]

    result = np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi

    return result
matrix_matrix_multiplicaton(U, V)
U.dot(V)


# Identity matrix
I = np.eye(3)
V.dot(I)

# Matrix inverse
Vs = V[[0,1,2]]
Vs
Vs_inv = np.linalg.inv(Vs)
Vs_inv
Vs_inv.dot(Vs)