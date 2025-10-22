# 使用方法

# 依赖项

- **pandas**: 用于数据处理和分析。
- **cv2**: OpenCV 库，用于图像处理。
- **matplotlib**: 用于数据可视化。
- **numpy**: 数值计算库。
- **torch**: PyTorch 框架，用于深度学习。
- **torchvision**: PyTorch 的计算机视觉工具包。
=======
## 使用方法

在项目根目录下创建两个文件夹：`rawdata` 和 `test`。在 `rawdata` 文件夹中创建两个子文件夹：`specs` 和 `pics`。请确保文件夹名称正确，注意大小写。

- `specs` 文件夹中存放光谱文件。
- `pics` 文件夹中存放照片。

### 配置 `config` 文件

```python
min_wavelength = 
max_wavelength = 
step_wavelength = 

# data_type = "IA" # Idea Optics
data_type = "OV" # Ocean View

epochs = 
delete_temp = 
```

- `min_wavelength`：起始波长。
- `max_wavelength`：终止波长。
- `step_wavelength`：波长步长。
- `data_type`：光谱数据类型，目前支持复享光学（Idea Optics）和海洋光学（Ocean View）的光谱仪数据。使用其中一个时，请注释掉另一个。
- `epochs`：机器学习的迭代次数。
- `delete_temp`：是否删除中间结果。如果设置为 `false`，中间结果将被保留；否则会自动删除。

### 运行项目

1. 安装项目依赖（见下文）。
2. 运行项目根目录下的 `run.py`。
3. 控制台会输出提示信息，请按照提示操作：
   - 首先会弹出一张图片，使用鼠标拖动选择滤光片的区域。每次选择后按回车确认，可以多次选择，直到按下 `ESC` 键结束。
   - 接着会弹出另一张图片，用于选择基底区域（无滤光片的区域）。此步骤只能选择一次。
4. 等待程序运行完成。

### 输出结果

结果将保存在 `result` 文件夹中，包括：
- 随机抽取的五份数据的真实值与预测值对比图。
- `rois.txt`：滤光片的坐标。
- `bgrois.txt`：基底区域的坐标。
- `model.pth`：训练好的模型。
- `input_mean.npy` 和 `input_std.npy`：输入归一化参数。
- `grey.csv`：灰度表。
- `specs.csv`：光谱表。

其中，`grey.csv` 和 `specs.csv` 是机器学习的输入数据。

### 测试模型

如果需要测试模型：
1. 将 `result` 文件夹中的 `rois.txt`、`bgrois.txt`、`model.pth`、`input_mean.npy` 和 `input_std.npy` 拷贝到 `test` 文件夹。
2. 修改 `test_grey.py` 中的参数。
3. 运行程序即可测试任意一张图片。

**注意**：如果输入维度不匹配，需要手动调整参数。目前该功能尚未完全实现。

## 项目依赖

本项目依赖以下主要库和工具：

### Python 库
- `torch`：用于深度学习模型的构建和训练。
- `numpy`：用于数值计算。
- `pandas`：用于数据处理和分析。
- `matplotlib`：用于数据可视化。
- `opencv-python`：用于图像处理。
- `shutil`：用于文件操作。

### 环境要求
- Python 版本：3.8 或更高。
- 操作系统：Windows 或 Linux。

### 安装依赖
运行以下命令安装所需依赖：
```bash
pip install torch numpy pandas matplotlib opencv-python
```
