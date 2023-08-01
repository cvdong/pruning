# @vector 
# 向量剪枝是一种非结构化的剪枝方法，它是将某些列和行上的参数设置为0，从而将参数的数量减少到原来的一部分

import numpy as np

np.random.seed(1)

def vector_pruning(matrix, idx):
    row, col = idx
    prune_matrix = matrix.copy()
    prune_matrix[row, :] = 0
    prune_matrix[:, col] = 0
    
    return prune_matrix

matrix = np.random.randn(3, 3)
idx  = (1, 1)

# prune the matrix
prune_matrix = vector_pruning(matrix, idx)
print(f"{matrix}\n\n")
print(f"{prune_matrix}")

# eg
import torch
import torch.nn as nn
from nn import Net

# 向量剪枝
net = Net()

for layer in net.modules():
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight.data.cpu().numpy()  # 获取卷积核权重张量的数据，并转为numpy数组
        num_filters, num_channels, filter_height, filter_width = weight.shape  # 获取卷积核的数量、通道数以及高度和宽度
        for i in range(num_filters):
            for j in range(num_channels):
                # 对每个卷积核的每个通道进行向量剪枝
                prune_idx = (1, 1)  # 剪枝行列索引
                weight[i, j] = vector_pruning(weight[i, j], prune_idx)  # 剪枝操作
        layer.weight.data = torch.from_numpy(weight).to(layer.weight.device)  # 将剪枝后的权重转为torch张量并赋给卷积层的权重

dummy_input = torch.randn(1, 3, 4, 4) # 构造一个形状为(1,3,4,4)的随机张量

with torch.no_grad():
    output = net(dummy_input)
print(output)
