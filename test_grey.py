import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    pic = "example.png"  # 替换为你的图片路径

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