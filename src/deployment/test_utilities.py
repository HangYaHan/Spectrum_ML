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

def flatten_spectrum(spectrum, display=False, peak_width=17, smooth_factor=0.95):
    """
    Flatten the spectrum by keeping the main peak and its shape, setting other values to 0.

    Args:
        spectrum (numpy.ndarray): The spectrum data (1D array).
        display (bool): Whether to display the spectrum before and after flattening.
        peak_width (int): The width of the peak to retain around the main peak.
        smooth_factor (float): A multiplier for the Gaussian smoothing. Higher values make the corners rounder.

    Returns:
        numpy.ndarray: The flattened spectrum.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def gaussian(x, mu, sigma):
        """Simple Gaussian function."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Find the index of the main peak
    peak_index = np.argmax(spectrum)

    # Create a new spectrum with all values set to 0
    flattened_spectrum = np.zeros_like(spectrum)

    # Retain the shape of the peak around the main peak
    start_index = max(0, peak_index - peak_width)
    end_index = min(len(spectrum), peak_index + peak_width + 1)
    flattened_spectrum[start_index:end_index] = spectrum[start_index:end_index]

    # Apply Gaussian smoothing to the peak edges
    sigma = (peak_width / 3) * smooth_factor  # Adjust sigma with smooth_factor
    for i in range(start_index, end_index):
        weight = gaussian(i, peak_index, sigma)
        flattened_spectrum[i] *= weight

    if display:
        # Plot the spectrum before and after flattening
        plt.figure(figsize=(10, 5))

        # Original spectrum
        plt.subplot(1, 2, 1)
        plt.plot(spectrum, label='Original Spectrum', color='blue')
        plt.title('Original Spectrum')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.legend()

        # Flattened spectrum
        plt.subplot(1, 2, 2)
        plt.plot(flattened_spectrum, label='Flattened Spectrum', color='red')
        plt.title('Flattened Spectrum')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return flattened_spectrum

def noise_spectrum(spectrum, noise_level=0.01, display=False):
    """
    Add Gaussian noise to the spectrum.

    Args:
        spectrum (numpy.ndarray): The original spectrum data (1D array).
        noise_level (float): The standard deviation of the Gaussian noise relative to the max value of the spectrum.
        display (bool): Whether to display the spectrum before and after adding noise.

    Returns:
        numpy.ndarray: The noisy spectrum.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    max_value = np.max(spectrum)
    noise = np.random.normal(0, noise_level * max_value, spectrum.shape)
    noisy_spectrum = spectrum + noise

    if display:
        # Plot the spectrum before and after adding noise
        plt.figure(figsize=(10, 5))

        # Original spectrum
        plt.subplot(1, 2, 1)
        plt.plot(spectrum, label='Original Spectrum', color='blue')
        plt.title('Original Spectrum')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.legend()

        # Noisy spectrum
        plt.subplot(1, 2, 2)
        plt.plot(noisy_spectrum, label='Noisy Spectrum', color='red')
        plt.title('Noisy Spectrum')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return noisy_spectrum

def rename_files_in_directory(directory, start_number=1, step=1):
    """
    Rename all files in the given directory, starting from `start_number` and incrementing by `step`,
    sorted by creation time.

    Args:
        directory (str): Path to the directory containing files to rename.
        start_number (int): Starting number for renaming.
        step (int): Step size for numbering.
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files_with_ctime = [(f, os.path.getctime(os.path.join(directory, f))) for f in files]
    files_with_ctime.sort(key=lambda x: x[1])  # Sort by creation time

    for idx, (filename, _) in enumerate(files_with_ctime):
        ext = os.path.splitext(filename)[1]  # Get file extension
        new_name = f"{start_number + idx * step}{ext}"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")


