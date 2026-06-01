import torch

# Crear dos matrices
A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

B = torch.tensor([
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0]
])

# Multiplicación de matrices
C = torch.matmul(A, B)

print("A:")
print(A)

print("B:")
print(B)

print("Resultado C = A x B:")
print(C)
