import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(1)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = 255 - img[i][j]
    return img


def accessBinary(img):
    img = accessPiexl(img)
    kernel = np.ones((3, 3), np.uint8)

    # img = cv2.medianBlur(img, 3)          # 均值滤波 即当对一个值进行滤波时，使用当前值与周围8个值之和，取平均做为当前值
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波 根据高斯的距离对周围的点进行加权,求平均值1，0.8， 0.6， 0.8
    # img = cv2.medianBlur(img, 3)          # 中值滤波 将9个数据从小到大排列，取中间值作为当前值

    # 腐蚀，去除边缘毛躁
    img = cv2.erode(img, kernel, iterations=1)

    # 二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -9)  # 自适应滤波   这个效果还行

    # 边缘膨胀
    img = cv2.dilate(img, kernel, iterations=8)
    return img


def findBorderContours(path, maxArea=100):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            if h > 20:
                border = [(x - 4, y - 4), (x + w + 4, y + h + 4)]  
                borders.append(border)
    borders.sort(key=lambda x: x[0][0])

    return borders


def showResults(path, borders, results=None):
    img = cv2.imread(path)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results is not None:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 2, (255, 250, 100), 2)
    return img


def transMNIST(path, borders, size=(28, 28), out_model=0):
    imgData = np.zeros((len(borders), size[0], size[0]), dtype='uint8')

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        h = abs(border[0][1] - border[1][1])

        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        h_extend = h // 5

        targetImg = cv2.copyMakeBorder(borderImg, h_extend, h_extend, int(extendPiexl * 1.1), int(extendPiexl * 1.1),
                                       cv2.BORDER_CONSTANT)

        targetImg = cv2.resize(targetImg, size)

        imgData[i] = targetImg 

    return imgData



def main_img(image_path, cnn):
    img = cv2.imread(image_path)

    borders = findBorderContours(image_path)
    imgData = transMNIST(image_path, borders)
    img1 = showResults(image_path, borders)

    result_number = []
    plt.figure()
    lie_num = len(borders) // 3 + 1
    for i in range(1, len(borders) + 1):
        plt.subplot(3, lie_num, i)  
        plt.imshow(imgData[i - 1])

        img = np.array(imgData[i - 1]).astype(np.float32)

        if (img[2][2] > 10) and (img[26][26] > 10) and (img[26][2] > 10) and (img[2][26] > 10):  # 如果周围是白底 就图像取反
            img = 256 - img

        img = np.expand_dims(img, 0)  # 扩展为，为[1,28,28]
        img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
        img = torch.from_numpy(img)  # 转成tensor

        test_output = cnn(img)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        result_number.append(int(pred_y))

    img2 = showResults(image_path, borders, result_number)
    cv2.waitKey(0)
    return result_number,img2


