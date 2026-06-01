import torch

# 1. The Integral Path
# This triggers the binary_kernel_reduce_vec with SIMD vectorization
int_tensor = torch.arange(1, 11, dtype=torch.int32)
print("------Integral sum------")
int_sum = torch.sum(int_tensor)

# 2. The Floating Point Path
# This triggers cascade_sum for numerical stability (Kahan/Pairwise)
float_tensor = torch.randn(100, 100, dtype=torch.float32)
print("------Float sum------")
float_sum = torch.sum(float_tensor)

print("------Results------")
print(f"Integer Sum (1-10): {int_sum.item()}")
print(f"Float Sum (Mean ~0): {float_sum.item():.4f}")

# You can also see the 'dim' reduction which uses the same underlying kernels
print("------Collapse------")
matrix = torch.ones((2, 3))
row_sum = torch.sum(matrix, dim=1)
print("------Result------")
print(f"\nRow-wise sum:\n{row_sum}")
