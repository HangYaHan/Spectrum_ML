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

    # !!!
    test_image = "23.png"

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
    output_dim = 701
    
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

    if config.flatten:
        output = test_utilities.flatten_spectrum(output, True)

    if config.add_noise:
        output = test_utilities.noise_spectrum(output, noise_level=0.001, display=False)

    # 将模型输出保存为 CSV 文件，文件名包含原始图像名
    output_csv_path = config.test_dir + f"model_output_{test_image.split('.')[0]}.csv"
    np.savetxt(output_csv_path, output, delimiter=",", fmt="%f")  # 按列保存，无表头
    print(f"Model output saved to {output_csv_path}")

    start_wl = float(config.min_wavelength)
    # 生成本样本的波长轴，长度与光谱向量一致
    length = len(output)  # 使用模型输出的长度
    sample_x = start_wl + np.arange(length) * float(config.step_wavelength)

    # 可视化输出光谱
    plt.figure(figsize=(10, 6))
    plt.plot(sample_x, output, label='Model Output Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Model Output Spectrum Visualization')
    plt.legend()
    plt.grid(False)
    plt.show()



