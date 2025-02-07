import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

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
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST标准化参数
    ])

    transform = transforms.Compose(transform_list)

    # 加载MNIST数据集
    full_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_train_data))  # 80% 用于训练
    val_size = len(full_train_data) - train_size  # 剩余 20% 用于验证
    
    train_data, val_data = random_split(full_train_data, [train_size, val_size])

    # 使用DataLoader来管理数据批次，增加num_workers来并行加载
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, val_loader
