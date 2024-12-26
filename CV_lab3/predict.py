import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义 CNN 模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 第一层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 第二层卷积
        self.pool = nn.MaxPool2d(kernel_size=2)       # 最大池化
        self.fc1 = nn.Linear(64 * 5 * 5, 128)         # 全连接层 1
        self.fc2 = nn.Linear(128, 10)                 # 全连接层 2（输出 10 类）

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义预处理函数
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((28, 28)),                 # 调整大小
    transforms.ToTensor(),                       # 转为张量
    transforms.Normalize((0.5,), (0.5,))         # 归一化
])

# 学号图片分割函数
def split_image(image, num_digits):
    """
    将包含多位数字的图片分割为单独的数字块。
    :param image: 输入图片（PIL 图像对象）。
    :param num_digits: 学号的位数。
    :return: 包含各数字的裁剪图像列表。
    """
    image = image.convert('L')  # 转为灰度图
    width, height = image.size
    digit_width = width // num_digits  # 每个数字的宽度
    digits = [image.crop((i * digit_width, 0, (i + 1) * digit_width, height)) for i in range(num_digits)]
    return digits

# 识别整串学号
def recognize_student_id_sequence(model_path, image_path, num_digits):
    """
    识别图片中的一串数字。
    :param model_path: 保存的模型路径。
    :param image_path: 学号图片路径。
    :param num_digits: 学号位数。
    :return: 识别出的学号字符串。
    """
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载并分割图片
    image = Image.open(image_path)
    digit_images = split_image(image, num_digits)

    # 逐一识别数字
    result = []
    for digit_image in digit_images:
        digit_image = transform(digit_image).unsqueeze(0).to(device)  # 数据预处理
        with torch.no_grad():
            output = model(digit_image)
            _, predicted = torch.max(output, 1)
            result.append(str(predicted.item()))

    return ''.join(result)

if __name__ == "__main__":
    # 模型文件路径和学号图片路径
    model_path = "cnn_model.pth"  # 之前保存的模型文件路径
    image_path = "input_image.jpg"  # 学号图片路径
    num_digits = 8  # 假设学号为 8 位数字

    # 调用识别函数
    result = recognize_student_id_sequence(model_path, image_path, num_digits)
    print(f"学号图片识别结果: {result}")
