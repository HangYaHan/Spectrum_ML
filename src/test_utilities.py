import config
import cv2
import os
import torch
from src import models
from src.models import ResNet1D

def get_rois(test_path):
    rois = []
    rois_path = os.path.join(test_path, "rois.txt")
    with open(rois_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            parts = line.split(',')
            if len(parts) == 4:  # 确保每行有 4 个值
                try:
                    rois.append(tuple(map(int, parts)))
                except ValueError:
                    print(f"Error parsing ROI: {line}")
            else:
                print(f"Invalid ROI format: {line}")
    return rois

def image_to_grey(image_name, rois, test_path):
    # Join the image path
    image_path = os.path.join(test_path, image_name)
    
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    greys = []
    for x, y, w, h in rois:
        # Crop the ROI region
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            greys.append(None)  # Or fill with a default value
        else:
            # Calculate the mean grayscale value
            greys.append(int(roi.mean()))
    
    return greys

def load_resnet1d_model(model_path, input_dim, output_dim, device='cpu'):
    """
    加载 ResNet1D 模型并返回实例。
    
    Args:
        model_path (str): 模型权重文件的路径。
        input_dim (int): 模型的输入维度。
        output_dim (int): 模型的输出维度。
        device (str): 运行设备（如 'cpu' 或 'cuda'）。
    
    Returns:
        torch.nn.Module: 加载的 ResNet1D 模型实例。
    """
    # 初始化模型
    model = ResNet1D(input_dim=input_dim, output_dim=output_dim)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 设置为评估模式
    model.eval()
    
    return model



