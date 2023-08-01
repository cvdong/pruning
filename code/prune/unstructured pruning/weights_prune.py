'''
# @weights
# 细粒度剪枝是指针对神经网络的每个权重进行剪枝，
# 相对于结构剪枝而言，细粒度剪枝不会改变神经网络的结构。
# 细粒度剪枝通过移除不重要的权重，可以达到减小神经网络模型大小的目的，从而提高模型的运行速度和存储效率。

'''
from nn import Net
import torch
import torch.nn as nn
import numpy as np

def prune_conv_layer(layer, prune_rate):
    """
    按比例裁剪卷积层的权重
    """
    if isinstance(layer, nn.Conv2d):
        weight = layer.weight.data.cpu().numpy() # 获取卷积核权重张量的数据，并转为numpy数组
        # print(weight.shape)
        # input_c * output_c * kernel_s_w * kernel_s_h
        num_weights = weight.size # 获取权重的总数量
        print(num_weights)
        num_prune = int(num_weights * prune_rate) # 计算需要裁剪的权重数量
        flat_weights = np.abs(weight.reshape(-1)) # 展开并取绝对值
        threshold = np.sort(flat_weights)[num_prune] # 找到需要保留的权重的最小阈值
        weight[weight < threshold] = 0 # 将小于阈值的权重置为0
        layer.weight.data = torch.from_numpy(weight).to(layer.weight.device) # 将剪枝后的权重转为torch张量并赋给卷积层的权重

net = Net()
prune_rate = 0.2 # 裁剪比例
for layer in net.modules():
    prune_conv_layer(layer, prune_rate)

dummy_input = torch.randn(1, 3, 4, 4) # 构造一个形状为(1,3,4,4)的随机张量

with torch.no_grad():
    output = net(dummy_input)
print(output)