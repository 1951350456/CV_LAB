import torch
import inference
import cv2

def main():
    # 指定图片路径
    file_path = input("请输入图片路径: ")
    if not file_path:
        print("未提供图片路径，程序退出。")
        return

    try:
        # 加载模型
        model_path = 'lyz.pkl'
        cnn = inference.CNN()
        cnn.load_state_dict(torch.load(model_path))
        cnn.eval()

        # 预测数字串
        result_num, img = inference.main_img(file_path, cnn)

        # 保存结果图像
        output_image_path = 'result.jpg'
        cv2.imwrite(output_image_path, img)

        # 打印识别结果
        result_string = ''.join(map(str, result_num))
        print(f"识别结果: {result_string}")
        print(f"处理后的图片已保存为: {output_image_path}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
