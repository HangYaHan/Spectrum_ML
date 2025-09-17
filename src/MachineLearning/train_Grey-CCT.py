import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        residual = self.fc1(x)
        out = self.relu(residual)
        out = self.fc2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    config.print_csv_shape(os.path.join(config.grey_csv_folder, "grey.csv"))
    config.print_csv_shape(os.path.join(config.spec_csv_folder, "CCT.csv"))

    grey = pd.read_csv('ProcessedData/FinalCSV/grey.csv').values  # shape: [2245, 18]
    cct = pd.read_csv('ProcessedData/FinalCSV/CCT.csv').values    # shape: [2245, 1]

    X = torch.tensor(grey, dtype=torch.float32)
    y = torch.tensor(cct, dtype=torch.float32)

    # 训练流程
    model = ResidualMLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # 从测试集中随机选取5个样本进行误差可视化
    import matplotlib.pyplot as plt

    # 假设全部数据用于训练，这里模拟划分测试集
    test_indices = random.sample(range(X.shape[0]), 5)
    X_test = X[test_indices]
    y_test = y[test_indices]

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()
        y_true = y_test.squeeze().numpy()
        errors = np.abs(y_pred - y_true)

    # 输出预测值、真实值和误差
    for i in range(5):
        print(f"Sample {i+1}: Predicted={y_pred[i]:.2f}, True={y_true[i]:.2f}, Error={errors[i]:.2f}")

    # 可视化误差
    plt.figure(figsize=(8, 4))
    plt.bar(range(5), errors)
    plt.xticks(range(5), [f"Sample {i+1}" for i in range(5)])
    plt.ylabel("Absolute Error")
    plt.title("Prediction Error on 5 Random Test Samples")
    plt.show()