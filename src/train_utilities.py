import os
import sys
import torchvision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from .models import ResNet1D

def train_resnet18_spectrum(
    min_wavelength,
    max_wavelength,
    step,
    src_csv,
    target_csv,
    save_path,
    epochs=1000,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu",
    random_seed=42,
    optimizer="adam",
    loss_fn="mse"
):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    import random

    # 自动创建保存目录，避免保存图片时报错
    os.makedirs(save_path, exist_ok=True)
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 数据集定义
    class SpectrumDataset(Dataset):
        def __init__(self, src_csv, target_csv):
            src_df = pd.read_csv(src_csv)
            tgt_df = pd.read_csv(target_csv)
            src_data = src_df.iloc[:, 1:].values.astype(np.float32)
            tgt_data = tgt_df.iloc[:, 1:].values.astype(np.float32)
            self.mean = src_data.mean(axis=0)
            self.std = src_data.std(axis=0)
            self.std[self.std == 0] = 1  # 防止除零
            self.X = ((src_data - self.mean) / self.std).astype(np.float32)
            self.y = tgt_data
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    # 数据加载
    dataset = SpectrumDataset(src_csv, target_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    output_dim = dataset.y.shape[1] if len(dataset.y.shape) > 1 else 1
    model = ResNet1D(input_dim, output_dim).to(device)

    # 损失函数
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss_fn: {loss_fn}")

    # 优化器
    if optimizer == "adam":
        optim_fn = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optim_fn = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    import matplotlib.pyplot as plt
    loss_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optim_fn.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optim_fn.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    print("训练完成！")

    # 绘制训练loss曲线
    plt.figure()
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_loss_curve.png'))
    plt.close()

    # 随机抽取5个测试样本，绘制预测值与真实值
    # 简单划分：最后10%为测试集
    test_size = max(5, int(len(dataset)*0.1))
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    test_indices = indices[-test_size:]
    # 确保抽样数量不超过测试集大小
    sample_size = min(5, len(test_indices))
    sample_indices = np.random.choice(test_indices, size=sample_size, replace=False)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            X_sample = torch.tensor(dataset.X[idx]).unsqueeze(0).to(device)
            y_true = dataset.y[idx]
            y_pred = model(X_sample).cpu().numpy().flatten()
            wavelengths = np.arange(min_wavelength, max_wavelength + 1, step)
            # 绘制原始光谱和生成光谱的对比图
            plt.figure()
            plt.plot(wavelengths, y_true, label='Original Spectrum', marker='o')
            plt.plot(wavelengths, y_pred, label='Generated Spectrum', marker='x')
            plt.title(f'Sample {i+1} Spectrum Comparison')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f'sample_{i+1}_spectrum_comparison.png'))
            plt.close()
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    # 保存输入数据归一化参数
    np.save(os.path.join(save_path, "input_mean.npy"), dataset.mean)
    np.save(os.path.join(save_path, "input_std.npy"), dataset.std)
    print("已保存归一化参数：input_mean.npy, input_std.npy")

if  __name__ == "__main__":
    pass