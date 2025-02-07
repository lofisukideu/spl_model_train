# utils.py
import torch

def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# def evaluate(model, test_loader):
#     model.eval()  # 设置模型为评估模式
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data.view(data.size(0), -1))  # 展平数据
#             _, predicted = torch.max(output, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()

#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy*100:.2f}%")
    
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct