import torch

# A 2x3 matrix (2 rows, 3 columns)
x = torch.tensor([[1, 1, 1],
                  [2, 2, 2]])

# Sum along dim 0 (squash the rows together)
# Result: [1+2, 1+2, 1+2] -> [3, 3, 3]
sum_dim0 = torch.sum(x, dim=0)

# Sum along dim 1 (squash the columns together)
# Result: [1+1+1, 2+2+2] -> [3, 6]
sum_dim1 = torch.sum(x, dim=1)

print(f"Original:\n{x}")
print(f"Sum dim=0: {sum_dim0}")
print(f"Sum dim=1: {sum_dim1}")
