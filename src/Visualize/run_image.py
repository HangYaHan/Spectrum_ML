import pandas as pd
import os
import sys

from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import torch
import torchvision.models as models

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn

    # 读取灰度数据
    grey_csv_path = os.path.join('ProcessedData', 'FinalCSV', 'grey_amplified.csv')
    df = pd.read_csv(grey_csv_path, header=None)

    # 定义与训练时一致的ResNet1D结构
    class BasicBlock1D(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )
        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class ResNet1D(nn.Module):
        def __init__(self, input_dim, output_dim, base_channels=32, layers=[2,2,2]):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.base_channels = base_channels
            self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm1d(base_channels)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(base_channels, base_channels, layers[0])
            self.layer2 = self._make_layer(base_channels, base_channels*2, layers[1], stride=2)
            self.layer3 = self._make_layer(base_channels*2, base_channels*4, layers[2], stride=2)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(base_channels*4, output_dim)
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = [BasicBlock1D(in_channels, out_channels, stride)]
            for _ in range(1, blocks):
                layers.append(BasicBlock1D(out_channels, out_channels))
            return nn.Sequential(*layers)
        def forward(self, x):
            x = x.unsqueeze(1)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.gap(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    # 构建模型并加载参数
    input_dim = df.shape[1] - 1  # 除去序号
    # output_dim 可根据训练时的 target_csv 列数确定，这里假设与训练一致
    # 获取 output_dim（spectrum.csv 列数减1）
    spectrum_csv_path = os.path.join('ProcessedData', 'FinalCSV', 'spectrum.csv')
    spec_df = pd.read_csv(spectrum_csv_path, header=None)
    output_dim = spec_df.shape[1] - 1
    model = ResNet1D(input_dim, output_dim)
    model_path = os.path.join('ProcessedData', 'FinalCSV', 'resnet18_spectrum', 'model.pth')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully:", type(model))

    output_dir = os.path.join('ProcessedData', 'FinalCSV', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 获取波长范围
    min_wavelength = 800
    max_wavelength = 1050
    step = 1
    wavelengths = np.arange(min_wavelength, max_wavelength + 1, step)

    for idx, row in df.iterrows():
        seq = row.iloc[0]  # 第一列序号
        input_data = row.iloc[1:].values.astype(np.float32)
        print(f"Seq {seq} input: {input_data}")
        input_tensor = torch.tensor(input_data).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        output_np = output.squeeze().cpu().numpy()
        print(f"Seq {seq} output: {output_np}")
    # 绘制输出，横坐标为波长
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, output_np, marker='o', linewidth=2)
    plt.title(f'Model Output for Seq {seq}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Value')
    # 自动设置y轴范围，突出曲线差异
    margin = (output_np.max() - output_np.min()) * 0.1
    plt.ylim(output_np.min() - margin, output_np.max() + margin)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'output_{int(seq):03d}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()



