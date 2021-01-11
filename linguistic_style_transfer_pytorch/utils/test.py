import torch
import torch.nn as nn


input = torch.randn(3)
m = nn.Sigmoid(input)
target = torch.empty(3).random_(2)

loss = nn.BCELoss()
output = loss(m, target)