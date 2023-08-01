# simple nn

import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True) # ReLU激活函数，inplace=True表示直接修改输入的张量，而不是返回一个新的张量
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv(3, 64, kernel_size=3, padding=1)
        self.conv2 = Conv(64, 64, kernel_size=3, padding=1)
        self.conv3 = Conv(64, 128, kernel_size=3, padding=1)
        self.conv4 = Conv(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.conv1(x) # 第一层卷积
        x = self.conv2(x) # 第二层卷积
        x = self.conv3(x) # 第三层卷积
        x = self.conv4(x) # 第四层卷积
        x = x.view(x.size(0), -1) # 展平
        x = self.fc1(x) # 第一个全连接层
        x = self.fc2(x) # 第二个全连接层
        return x