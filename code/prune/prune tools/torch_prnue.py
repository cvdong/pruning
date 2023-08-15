# torch.nn.utils.prune

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Define a simple linear layer
class MyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(MyLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.t())
    
# Create an instance of the linear layer
linear_layer = MyLinearLayer(5, 3)
print("before pruning")
print(linear_layer.weight)

# Apply pruning to the layer's weight
prune.random_unstructured(linear_layer, name='weight', amount=0.5)

# Define the forward pre-hook
def apply_pruning(module, input):
    module.weight.data = module.weight * module.weight_mask

# Register the forward pre-hook
linear_layer.register_forward_pre_hook(apply_pruning)

# Perform a forward pass
input_tensor = torch.randn(1, 5)
output_tensor = linear_layer(input_tensor)

print("after pruning")

print("Input Tensor:")
print(input_tensor)

print("Weight Tensor")
print(linear_layer.weight)

print("Output Tensor:")
print(output_tensor)