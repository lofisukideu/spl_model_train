import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from model import LighterCNN  # 确保该模型类路径正确
import os
import time

# 设备选择 (使用 GPU 加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **加载训练好的模型**
def load_model(model_path='best_model.pth'):
    model = LighterCNN().to(device)  # 载入模型并移动到设备
    try:
        checkpoint = torch.load(model_path, map_location=device)  # 适配 CPU/GPU
        model.load_state_dict(checkpoint, strict=False)  # 允许部分参数不匹配
        model.eval()  # 设置为评估模式
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 发生错误: {e}\n请检查模型路径是否正确！")
        exit()
    return model

# **图像预处理**
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 转为灰度
    
    img = img.resize((28, 28), Image.BILINEAR)  # 调整尺寸
    
    enhancer = ImageEnhance.Contrast(img)
    
    enhanced_img = enhancer.enhance(factor=2.0)
    
    # **优化缩放方式，避免失真**
    transform = transforms.Compose([
        transforms.CenterCrop(28),  # 避免比例失真
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化
    ])

    img_tensor = transform(enhanced_img).unsqueeze(0).to(device)  # 转换为 Tensor，并移动到设备
    return img_tensor

# **进行预测，并返回 Softmax 置信度**
def predict_with_confidence(model, img_tensor):
    model.eval()
    
    with torch.no_grad():  # 关闭梯度计算，提高推理速度
        output = model(img_tensor)  
        probabilities = F.softmax(output, dim=1)  # 计算 softmax 概率

        predicted_class = torch.argmax(probabilities, dim=1).item()  # 获取最大概率的类别
        confidence = probabilities[0][predicted_class].item()  # 最大类别的置信度
        class_confidences = probabilities.squeeze().cpu().tolist()  # 转为 Python 列表
    return predicted_class, confidence, class_confidences

# **显示输入图像**
def show_image(image_path):
    img = Image.open(image_path).convert('L')  # 打开并转为灰度图
    img = img.resize((28, 28))  # 调整尺寸
    plt.imshow(np.array(img), cmap='gray')  # 显示图像
    plt.axis("off")  
    plt.title("输入图片")
    plt.show()

# **显示 Softmax 置信度分布**
def show_confidence_distribution(class_confidences, predicted_class):
    plt.bar(range(10), class_confidences, color=['blue' if i != predicted_class else 'red' for i in range(10)])
    plt.xticks(range(10))
    plt.xlabel("类别")
    plt.ylabel("置信度")
    plt.title("Softmax 置信度分布")
    plt.show()
    
# **✅ 保存预测的图片**
def save_prediction_image(image_path, predicted_digit, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
    
    # 读取并调整图片
    img = Image.open(image_path).convert('L').resize((28, 28))
    
    # 生成唯一文件名（防止覆盖）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(output_dir, f"pred_{predicted_digit}_{timestamp}.png")
    
    img.save(save_path)
    print(f"✅ 预测的图片已保存: {save_path}")

# **主循环**
def main():
    model = load_model()  # 加载训练好的模型
    print("\n✅ 模型加载完成，输入图片路径开始预测，输入 'exit' 退出程序。")

    while True:
        # 获取用户输入的图片路径
        image_path = input("\n📷 请输入手写数字图片的路径 (输入 'exit' 退出)： ")
        if image_path.lower() == 'exit':
            print("👋 程序已退出。")
            break

        # **检查路径是否存在**
        if not os.path.exists(image_path):
            print(f"❌ 错误: 文件 '{image_path}' 不存在，请检查路径！")
            continue

        try:
            # 处理图像
            img_tensor = preprocess_image(image_path)

            # 进行预测
            predicted_digit, confidence, class_confidences = predict_with_confidence(model, img_tensor)

            # 输出预测结果
            print(f"\n🎯 预测的数字是: {predicted_digit}，置信度: {confidence:.2f}")
            for i, class_confidence in enumerate(class_confidences):
                print(f"类别 {i}: {class_confidence * 100:.2f}%")

            # **显示输入图像和 Softmax 置信度分布**
            show_image(image_path)
            show_confidence_distribution(class_confidences, predicted_digit)

            # **✅ 保存当前输入图片，并带预测标签**
            save_prediction_image(image_path, predicted_digit)

        except Exception as e:
            print(f"❌ 发生错误: {e}\n请检查图片格式是否正确！")

if __name__ == "__main__":
    main()
