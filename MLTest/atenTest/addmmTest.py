import torch

# Matriz que se va a sumar
input = torch.tensor([
    [1.0, 1.0],
    [1.0, 1.0]
])

# Primera matriz
A = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("A:", A)

# Segunda matriz
B = torch.tensor([
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0]
])

print("B:", B)

# addmm
C = torch.addmm(input, A, B)

print("Resultado:")
print(C)
