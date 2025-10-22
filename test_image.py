import torch
import numpy as np
import matplotlib.pyplot as plt
from src import test_utilities
import config
import pandas as pd

if __name__ == "__main__":
    print("This is the program to test input image into model and get output spectrum.")

    model_path = config.test_dir + "model.pth"
    input_mean_path = config.test_dir + "input_mean.npy"
    input_std_path = config.test_dir + "input_std.npy"
    test_image = "1.bmp"

    # 获取灰度值
    rois = test_utilities.get_rois(config.test_dir)
    greys = test_utilities.image_to_grey(test_image, rois, config.test_dir)

    # 打开 bgrois.txt 并读取第五个数 x
    bgrois_path = config.test_dir + "bgrois.txt"
    with open(bgrois_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split(',')
        if len(parts) < 5:
            raise ValueError(f"Invalid format in bgrois.txt: {line}")
        try:
            x = float(parts[4])
        except ValueError:
            raise ValueError(f"Invalid x value in bgrois.txt: {parts[4]}")

    if x == 0:
        raise ValueError("x value in bgrois.txt is zero, cannot divide by zero.")

    # 将获取的灰度值除以 x
    greys = [grey / x for grey in greys]

    print("Greys (normalized):", greys)

    input_dim = rois.__len__()
    output_dim = 255
    
    # 加载模型
    model = test_utilities.load_resnet1d_model(model_path, input_dim, output_dim)
    print(model)

    # 加载归一化参数
    input_mean = np.load(input_mean_path)
    input_std = np.load(input_std_path)

    # 对灰度值进行归一化
    greys = np.array(greys, dtype=np.float32)
    normalized_greys = (greys - input_mean) / input_std

    # 转换为张量并输入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = torch.tensor(normalized_greys, dtype=torch.float32).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 输出结果
    output = output.cpu().numpy().flatten()
    print("Model Output:", output)

    # 将模型输出保存为 CSV 文件
    output_csv_path = config.test_dir + "model_output.csv"
    np.savetxt(output_csv_path, output, delimiter=",", fmt="%f")  # 按列保存，无表头
    print(f"Model output saved to {output_csv_path}")

    # 读取 test_dir 下的 1.csv 文件，跳过第一行
    csv_path = config.test_dir + "1.csv"
    csv_data = pd.read_csv(csv_path, header=None, usecols=[0, 1], names=["Wavelength", "Value"], skiprows=1)  # 跳过表头行

    # 确保模型输出的 x 轴范围与 CSV 数据的 x 轴范围一致
    csv_x = csv_data["Wavelength"].astype(float).values  # 转换为浮点数
    model_x = np.linspace(csv_x.min(), csv_x.max(), len(output))

    # 绘制模型输出和 CSV 数据图表
    plt.figure()
    plt.plot(model_x, output, label="Model Output")
    plt.plot(csv_x, csv_data["Value"], label="CSV Data", linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Value")
    plt.title("Model Output vs CSV Data")
    plt.legend()
    plt.grid()
    plt.show()



