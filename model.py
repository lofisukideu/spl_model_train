import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        
        # 适当减少通道数，避免过拟合
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # 使用全局平均池化代替 flatten
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 减少全连接层的大小
        # self.fc1 = nn.Linear(64, 128)  # 直接输入 64 维特征
        # self.fc2 = nn.Linear(128, 10)  # 输出类别
        self.fc1 = nn.Linear(64, 64)  # 直接减少参数
        self.fc2 = nn.Linear(64, 10)
        
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x)  # 使用全局平均池化
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x