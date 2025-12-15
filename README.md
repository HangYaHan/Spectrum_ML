## 项目简介

Spectrum_ML 是一个将滤光片图片映射到光谱的端到端流水线，包含数据清洗、ROI 交互标注、特征提取与基于 PyTorch 的 1D ResNet 训练与推理。项目同时兼容 OceanView (OV) 与 Idea Optics (IA) 两类光谱数据格式。

## 目录速览

- 配置与入口：[config.py](config.py)、[run.py](run.py)
- 数据处理： [src/dataprocess_pipeline.py](src/dataprocess_pipeline.py)、[src/dataprocess_utilities.py](src/dataprocess_utilities.py)
- 训练： [src/train_utilities.py](src/train_utilities.py)、[src/models.py](src/models.py)
- 推理与工具：test_image.py、test_camera.py、test_grey.py、merge_image.py
- 数据与结果：rawdata/（输入） 、result/（输出）、temp/（中间件，可选保留）

## 环境准备

- Python 3.12（建议使用 venv 或 conda）。
- 联网安装：
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
- 离线安装（仓库含 packages/）：
```powershell
python -m pip install --no-index --find-links=packages -r requirements.txt
```

## 配置说明（config.py）

| 参数                            | 作用                               | 建议值           |
| ------------------------------- | ---------------------------------- | ---------------- |
| source_folder                   | 原始数据根目录                     | .\rawdata        |
| temp_folder                     | 处理时的临时目录                   | .\temp           |
| target_folder                   | 输出前缀目录（运行时自动加时间戳） | .\result\result  |
| min_wavelength / max_wavelength | 截取光谱范围                       | 400 / 1100       |
| step_wavelength                 | 波长步长（用于生成波长轴）         | 1                |
| data_type                       | 数据来源：OV 或 IA                 | OV 或 IA         |
| delete_temp                     | 流水线结束后是否清理 temp          | True / False     |
| epochs                          | 训练轮数                           | 1000+ (根据需要) |
| flatten / add_noise             | 推理时单色光还原或增强鲁棒性可选项 | False            |

若使用 IA 数据，请将 data_type 设为 IA；否则默认 OV。波长范围需与光谱文件的分辨率匹配。

## 数据准备

- rawdata/specs/：放置光谱文件（OV 使用 .txt，IA 使用 .csv）。
- rawdata/pics/：放置对应的滤光片图像（bmp/jpg/png，命名无严格要求，程序会按修改时间重命名并对齐）。
- 可选：在 rawdata 下预先放置 rois.txt 与 bgrois.txt，可跳过交互式标注。

## 一键训练（OV/IA 通用）

入口脚本 [run.py](run.py) 会按时间戳创建输出目录并调用对应流水线后训练 1D ResNet。

```powershell
python .\run.py
```

流程要点：
- 复制 rawdata 到 temp 并重命名、对齐图片与光谱。
- OV：去头、取整、按波长截取并合并为 spectrum.csv；IA：去头后合并为单一 spectrum.csv。
- 交互式选择滤光片 ROI 与背景 ROI，生成 rois.txt 与 bgrois.txt；随后计算灰度生成 grey.csv。
- 训练 ResNet1D，输出模型、归一化参数与训练曲线到 result/result_时间戳/。
- delete_temp=True 时会清理 temp；否则保留排查。

## 推理与可视化

- 单张图片推理：配置 test_image.py 中模型路径、归一化文件、待测图片后运行：
```powershell
python .\test_image.py
```
输出 CSV 与光谱对比图保存到 test/。

- 相机实时/灰度测试：可参考 test_camera.py、test_grey.py。

- 多结果合并：在 merge/ 下放置待合并 csv/txt，配置 merge_image.py 后运行：
```powershell
python .\merge_image.py
```

## 产出物

- result/result_时间戳/：训练模型 model.pth、归一化参数 input_mean.npy / input_std.npy、训练曲线 training_loss_curve.png、若干样本对比图 sample_*_spectrum_comparison.png、grey.csv、spectrum.csv。
- test/：推理结果 csv 与可视化图。

## 常见问题

- pip 找不到：在 PowerShell 使用 python -m pip，或将 Python 安装目录下 Scripts 加入 PATH。
- 文件数不匹配：若 pics 与 specs 数量不同，流水线会直接报错，请确认输入数据。
- 光谱长度不一致：检查 step_wavelength 与原始分辨率是否匹配，或输入文件是否缺失数据。
- ROI 选择窗口无法弹出：确保本机已安装 GUI 依赖（OpenCV）并在桌面环境运行。
