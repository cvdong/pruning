from remove_prune import BigModel, train
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1.load a model and inspect it
model = BigModel()
model.load_state_dict(torch.load("C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/r_big_model.pth"))

# # code snippest for inspecting the structure
# for name, module in model.named_modules():
#     print(name, module)
#     pass

# 2. get threshold based on l1 norm for each layer
l1norms_for_local_threshold = []

for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        l1norm_buffer_name = f"{name}_l1norm_buffer"
        l1norm = getattr(model, l1norm_buffer_name)
        l1norms_for_local_threshold.append(l1norm)
 
T_conv1 = torch.sort(l1norms_for_local_threshold[0])[0][int(len(l1norms_for_local_threshold[0])*0.5)]

# 3. prune the conv1's outchannel based on l1 norm (axis = 0)
conv1 = model.conv1  # [32x1x3x3]
conv2 = model.conv2  # [16x32x3x3]

conv1_l1norm_buffer = model.conv1_l1norm_buffer
conv2_l1norm_buffer = model.conv2_l1norm_buffer

# Top conv
keep_idxs = torch.where(conv1_l1norm_buffer >= T_conv1)[0]
k = len(keep_idxs)

conv1.weight.data = conv1.weight.data[keep_idxs]
conv1.bias.data = conv1.bias.data[keep_idxs]
conv1_l1norm_buffer.data = conv1_l1norm_buffer.data[keep_idxs]
conv1.out_channels = k # 16

# Bottom conv
_, keep_idxs = torch.topk(conv2_l1norm_buffer, k)

conv2.weight.data = conv2.weight.data[:,keep_idxs]
conv2_l1norm_buffer.data = conv2_l1norm_buffer.data[keep_idxs]
conv2.in_channels = k # 16

# code snippest for inspecting the structure
for name, module in model.named_modules():
    print(name, module)
    pass

# Save the pruned model state_dict
torch.save(model.state_dict(), "C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/r_pruned_model.pth")

# dummy input
dummy_input = torch.randn(1, 1, 28, 28)

# export to onnx
torch.onnx.export(model, dummy_input, "C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/r_pruned_model.onnx")

#################################### FINE TUNE ######################################
# Prepare the MNIST dataset
model.load_state_dict(torch.load("C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/r_pruned_model.pth"))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
big_model = train(model, train_loader, criterion, optimizer, device='cuda', num_epochs=10)

# Save the trained big network
torch.save(model.state_dict(), "C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/r_pruned_model_after_finetune.pth")

#  BigModel(
#   (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (fc): Linear(in_features=12544, out_features=10, bias=True)
# )
# conv1 Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# conv2 Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# fc Linear(in_features=12544, out_features=10, bias=True)
# Epoch 1, Loss: 0.12255196460000953
# Epoch 2, Loss: 0.0337646949725019
# Epoch 3, Loss: 0.02136572280728411