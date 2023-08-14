import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义网络模型
class BigModel(nn.Module):
    def __init__(self):
        super(BigModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 加载模型和测试数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST('C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
model = BigModel()
model.load_state_dict(torch.load("C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/finetuned_pruned_model.pth"))

# 3. 测试模型并计算准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        # if i == 10:
        #     break  # 只测试前10个batch的数据
        outputs = model(inputs.view(inputs.size(0), -1))
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        if i == 1:
            # 可视化第一个batch的数据
            fig, axs = plt.subplots(2, 5)
            axs = axs.flatten()
            for j in range(len(axs)):
                axs[j].imshow(inputs[j].squeeze(), cmap='gray')
                axs[j].set_title(f"Target: {targets[j]}, Predicted: {predicted[j]}")
                axs[j].axis('off')
            # plt.savefig("fine-tune.png", bbox_inches="tight")
            plt.show()


accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

# 模型参数量
from torchsummary import summary
summary(model.to("cuda"), input_size=(512, 784), device="cuda")