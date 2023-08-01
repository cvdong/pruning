'''
# @kernel
卷积核剪枝是非结构化剪枝的一种形式，它是指对卷积神经网络中的卷积核进行剪枝，
即删除某些卷积核中的权重，以达到减少模型参数数量、减少计算量、提高模型运行效率等目的的一种技术。
'''
import torch.nn as nn
import numpy as np
import torch

def prune_conv_layer(layer, prune_rate):
    """
    对卷积层进行剪枝,将一定比例的权重设置为0
    """
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight.data.cpu().numpy() # 去到当前层的卷积核权重 ===> [128,64,3,3]
        num_weights = weight.size
        num_prune = int(num_weights * prune_rate)
        # 计算每个filter的L2范数
        norm_per_filter = np.sqrt(np.sum(weight**2, axis=(1, 2, 3)))
        # 根据L2范数排序，选择剪枝比例最高的一定数量的卷积核
        indices = np.argsort(norm_per_filter)[:num_prune]
        # 将这些kernel中的所有权重置为0
        weight[indices] = 0
        layer.weight.data = torch.from_numpy(weight).to(layer.weight.device)

import numpy as np

# 构造4个1x3x3的filter
filter1 = np.array([[[0, 5, 2],
                     [3, 9, 10],
                     [6, 6, 14]]])

filter2 = np.array([[[5, 6, 8],
                     [3, 4, 0],
                     [0, 6, 12]]])

filter3 = np.array([[[2, 10, 9],
                     [7, 11, 5],
                     [0, 12, 5]]])

filter4 = np.array([[[6, 2, 8],
                     [9, 3, 8],
                     [4, 9, 3]]])

# 将4个filter拼接成一个卷积核
weight = np.stack([filter1, filter2, filter3, filter4], axis=0)

# 定义剪枝比例和要剪枝的数量
prune_rate = 2/3 # 要去掉这么多
num_prune = int(weight.shape[0] * prune_rate)

# 计算每个filter的L2范数
# 范数(Norm)是向量空间的一种函数，其用来衡量向量的大小。
# 在数学上，范数是一种将向量映射到非负实数的函数，范数有很多种，
# 例如L1数、L2范数等。其中L2范数又称为欧几里得范数，它是指向量各元素的平方和的平方根，

norm_per_filter = np.sqrt(np.sum(weight ** 2, axis=(1, 2, 3)))
print()
print(f"每个卷积核的L2范数: {norm_per_filter}")

# 根据L2范数排序，选择剪枝比例最高的一定数量的卷积核
indices = np.argsort(norm_per_filter)[:num_prune]
print(f"需要剪枝的filter索引: {indices}")

# 将这些filter中的所有权重置为0
weight[indices] = 0
print(f"剪枝后的权重矩阵: \n {weight}")