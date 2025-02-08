import torch
import torch.nn as nn
import torch.nn.functional as F

class LighterCNN(nn.Module):
    def __init__(self):
        super(LighterCNN, self).__init__()
        
        # 进一步减少通道数
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # 只使用两次池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 直接减少全连接层
        self.fc = nn.Linear(32, 10) 
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层不池化
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 第三层池化

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # 直接分类
        return x
