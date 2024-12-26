import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl

def zh_ch(string):
    return string.encode("gb2312").decode(errors="ignore")

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def read_image(image_path):
    """
    从指定路径读取图像并转换为灰度图像。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法在指定路径找到图像。")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, grayscale_image

def sobel_filter(image):
    """
    实现Sobel滤波，根据图中的公式计算Gx，Gy和G。
    """
    rows, cols = image.shape
    Gx_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    Gy_kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

    Gx = np.zeros_like(image, dtype=np.float32)
    Gy = np.zeros_like(image, dtype=np.float32)

    padded_image = np.pad(image, 1, mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+3, j:j+3]
            Gx[i, j] = np.sum(region * Gx_kernel)
            Gy[i, j] = np.sum(region * Gy_kernel)

    G = np.sqrt(Gx**2 + Gy**2)
    return np.clip(G, 0, 255).astype(np.uint8), np.clip(Gx, 0, 255).astype(np.uint8), np.clip(Gy, 0, 255).astype(np.uint8)

def compute_histogram(image):
    """
    手动计算并绘制图像的颜色直方图。
    """
    hist_b = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_r = np.zeros(256, dtype=int)

    rows, cols, _ = image.shape
    for i in range(rows):
        for j in range(cols):
            b, g, r = image[i, j]
            hist_b[b] += 1
            hist_g[g] += 1
            hist_r[r] += 1

    plt.plot(hist_b, color='b', label='Blue')
    plt.plot(hist_g, color='g', label='Green')
    plt.plot(hist_r, color='r', label='Red')
    plt.title("颜色直方图")
    plt.xlabel("像素值")
    plt.ylabel("频率")
    plt.legend()
    plt.show()

def extract_texture_features(image):
    """
    手动计算图像的维理特征（均值和标准差）。
    """
    rows, cols = image.shape
    pixel_sum = 0
    pixel_squared_sum = 0
    total_pixels = rows * cols

    for i in range(rows):
        for j in range(cols):
            pixel_value = image[i, j]
            pixel_sum += pixel_value
            pixel_squared_sum += pixel_value ** 2

    mean = pixel_sum / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std_dev = np.sqrt(variance)

    return np.array([mean, std_dev])

def save_features(features, filename):
    """
    将提取的维理特征保存为 .npy 文件。
    """
    np.save(filename, features)

def apply_filter(image, kernel):
    """
    使用卷积操作对图像应用指定的卷积核滤波。
    """
    rows, cols = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            filtered_value = np.sum(region * kernel)
            filtered_image[i, j] = filtered_value

    return np.clip(filtered_image, 0, 255).astype(np.uint8)



def main():
    # 输入图像路径（请替换为您的图像路径）
    image_path = "input_image.jpg"

    # 读取图像
    original_image, grayscale_image = read_image(image_path)

    # 给定的自定义卷积核
    custom_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])

    # 使用 Sobel 滤波
    sobel_filtered_image, sobel_Gx, sobel_Gy = sobel_filter(grayscale_image)

    # 使用自定义卷积核进行滤波
    custom_filtered_image = apply_filter(grayscale_image, custom_kernel)

    # 保存滤波结果图像
    cv2.imwrite("sobel_filtered_image.png", sobel_filtered_image)
    cv2.imwrite("sobel_Gx.png", sobel_Gx)
    cv2.imwrite("sobel_Gy.png", sobel_Gy)

    # 计算直方图
    compute_histogram(original_image)

    # 提取维理特征
    sobel_texture_features = extract_texture_features(sobel_filtered_image)

    # 保存维理特征
    save_features(sobel_texture_features, "sobel_texture_features.npy")

    # 显示图像
    cv2.imshow(zh_ch("original image"), original_image)
    cv2.imshow(zh_ch("grayscale image"), grayscale_image)
    cv2.imshow(zh_ch("Sobel filtered image"), sobel_filtered_image)
    cv2.imshow(zh_ch("Custom Convolution Kernel Filtered Image"), custom_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
