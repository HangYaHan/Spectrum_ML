import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 替换为你的图片路径，支持 .png, .bmp, .jpg
    pic = "1.jpg"  # 替换为你的图片路径

    # 检查文件扩展名是否支持
    supported_formats = ['.png', '.bmp', '.jpg']
    if not any(pic.lower().endswith(ext) for ext in supported_formats):
        print(f"不支持的文件格式: {pic}")
        exit()

    # 读取图片
    img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法打开图片: {pic}")
        exit()

    # 计算灰度直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 绘制灰度直方图
    plt.figure()
    plt.title("灰度直方图")
    plt.xlabel("灰度值")
    plt.ylabel("像素数")
    plt.plot(hist, color='gray')
    plt.grid()
    plt.show()