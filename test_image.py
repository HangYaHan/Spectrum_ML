import torch
import numpy as np
import matplotlib.pyplot as plt
from src import test_utilities
import config

if __name__ == "__main__":
    print("This is the program to test input image into model and get output spectrum.")

    model_path = config.test_dir + "model.pth"
    input_mean_path = config.test_dir + "input_mean.npy"
    input_std_path = config.test_dir + "input_std.npy"
    test_image = "1000.bmp"

    # 获取灰度值
    rois = test_utilities.get_rois(config.test_dir)
    greys = test_utilities.image_to_grey(test_image, rois, config.test_dir)
    print("Greys:", greys)
    input_dim = rois.__len__()
    output_dim = config.max_wavelength - config.min_wavelength + 1
    

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

    # 绘制输出图表
    x_axis = np.arange(config.min_wavelength, config.min_wavelength + len(output))
    plt.figure()
    plt.plot(x_axis, output, label="Model Output")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Output Value")
    plt.title("Model Output Visualization")
    plt.legend()
    plt.grid()
    plt.show()


