import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import CNN  # 你定义的CNN模型类，确保此路径正确

# 加载训练好的模型
def load_model(model_path='model.pth'):
    model = CNN()  # 初始化模型
    checkpoint = torch.load(model_path)  # 加载模型参数
    model.load_state_dict(checkpoint, strict=False)  # 使用 strict=False 忽略不匹配的层
    model.eval()  # 设置为评估模式
    return model

# 图像预处理：将图像转换为与训练时一致的格式
def preprocess_image(image_path):
    # 打开图片并转为灰度图
    img = Image.open(image_path).convert('L')  
    # 将图片大小调整为28x28（MNIST数据集的标准尺寸）
    img = img.resize((28, 28))

    # 使用 transforms 进行预处理：转换为Tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 与训练时使用的标准化一致
    ])

    img_tensor = transform(img)  # 将图像转换为Tensor
    img_tensor = img_tensor.unsqueeze(0)  # 添加一个batch维度
    print(f"预处理后的图像形状: {img_tensor.shape}")  # 打印图像形状
    return img_tensor

# 使用模型进行数字预测
def predict(model, img_tensor):
    with torch.no_grad():  # 不需要计算梯度，节省内存
        output = model(img_tensor)  # 获取模型输出
        _, predicted_class = torch.max(output, 1)  # 获取最大概率的类别
    return predicted_class.item()  # 返回预测的数字类别

# 可视化输入图像
def show_image(image_path):
    img = Image.open(image_path).convert('L')  # 打开并转为灰度图
    img = img.resize((28, 28))  # 调整尺寸
    plt.imshow(np.array(img), cmap='gray')  # 显示图像
    plt.axis('off')  # 关闭坐标轴
    plt.show()

def main():
    # 输入手写数字图像路径
    image_path = input("请输入手写数字图片的路径: ")

    # 加载训练好的模型
    model = load_model()

    # 图像预处理
    img_tensor = preprocess_image(image_path)

    # 进行预测
    predicted_digit = predict(model, img_tensor)

    # 显示输入图像和预测结果
    show_image(image_path)
    print(f"预测的数字是: {predicted_digit}")

if __name__ == "__main__":
    main()
