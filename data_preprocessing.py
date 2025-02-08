import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

def load_data(batch_size=64, use_augmentation=False):
    # 数据增强
    transform_list = []
    if use_augmentation:
        transform_list.extend([
            transforms.RandomRotation(10),  # 随机旋转
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机仿射变换
        ])

    # 转换为 tensor 和标准化
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化参数
    ])

    transform = transforms.Compose(transform_list)

    # **加载完整 MNIST 数据集**
    full_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # **计算完整训练集的类别分布**
    full_targets = np.array(full_train_data.targets)  # 获取所有标签
    class_counts = np.bincount(full_targets)  # 统计类别数量
    num_samples = len(full_targets)  # 总样本数

    # **计算采样权重**（类别少 -> 采样概率高）
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for label in full_targets]

    # **创建 WeightedRandomSampler**
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

    # **在完整数据集上进行采样**
    full_train_loader = DataLoader(full_train_data, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # **使用 `random_split` 进行数据划分**
    train_size = int(0.8 * len(full_train_data))  # 80% 训练
    val_size = len(full_train_data) - train_size  # 20% 验证
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    # **创建 DataLoader**
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # **统计修正后的训练数据类别分布**
    all_labels = []
    for _, labels in full_train_loader:
        all_labels.extend(labels.numpy())

    print("训练数据类别分布（重新采样后）:", Counter(all_labels))

    return train_loader, test_loader, val_loader
