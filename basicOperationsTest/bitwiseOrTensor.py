import torch

# Crear dos tensores de ejemplo
a = torch.tensor([1, 2, 3], dtype=torch.int32)
b = torch.tensor([4, 5, 6], dtype=torch.int32)

# Operación bitwise OR (equivalente a aten.bitwise_or.Tensor)
resultado = torch.bitwise_or(a, b)

print("Tensor A:", a)
print("Tensor B:", b)
print("Resultado (bitwise OR):", resultado)
