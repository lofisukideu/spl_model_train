import torch
import torch.optim as optim
import torch.nn as nn
from model import OptimizedCNN
from data_preprocessing import load_data
from utils import calculate_accuracy, save_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 获取数据加载器
train_loader, test_loader, val_loader = load_data(batch_size=64)

# 模型初始化
model = OptimizedCNN()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 学习率调度器：当验证集精度没有提升时，降低学习率
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

# 训练模型
def train_model():
    model.train()
    best_val_acc = 0.0
    patience_counter = 0  # 记录验证精度未提升的轮数

    for epoch in range(10):  # 训练轮数
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计准确率
            correct += calculate_accuracy(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

        # 验证
        val_acc = validate_model()

        # 如果验证准确率没有提升，增加早停计数器
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 重置计数器
            # 保存当前最优模型
            save_model(model, 'best_model.pth')
            print("Best model saved")
        else:
            patience_counter += 1

        # 早停机制：若验证准确率在连续3个epoch内没有提升，停止训练
        if patience_counter >= 3:
            print(f"Early stopping after epoch {epoch+1}")
            break

        # 使用学习率调度器调整学习率
        scheduler.step(val_acc)
        # 打印当前学习率
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")


# 验证模型
def validate_model():
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_correct += calculate_accuracy(outputs, labels)
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")
    return val_acc

# 测试模型
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            correct += calculate_accuracy(outputs, labels)
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 训练和测试
if __name__ == "__main__":
    train_model()
    test_model()
