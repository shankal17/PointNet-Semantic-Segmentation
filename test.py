import torch
import torch.nn as nn

m = nn.BatchNorm1d(6)
input_tensor = torch.randn(4, 6)
print(f"input:\n {input_tensor}")
output = m(input_tensor)
print(f"output:\n {output}")

