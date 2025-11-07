# Spectrum_ML — 使用说明
这是一个用 PyTorch 与数据预处理管线训练/测试基于图像预测光谱的项目。下面是更详细的项目说明、安装与运行步骤（包含 Windows PowerShell 的注意事项）。

## 项目结构（重要文件）

- `config.py`：全局配置项（波长范围、路径、处理选项等）。
- `run.py`：主流程脚本（数据处理 + 训练/评估）。
- `test_image.py`：单张图片推理并可视化输出光谱。
- `test_grey.py` / `test_camera.py`：辅助测试脚本。
- `merge_image.py`：合并并可视化多个模型/测量结果。
- `requirements.txt`：Python 依赖清单。
- `src/`：项目源代码（`dataprocess_pipeline.py`、`dataprocess_utilities.py`、`test_utilities.py`、`models.py` 等）。
- `rawdata/`：原始数据目录（`pics/`、`specs/`）。
- `result/`：训练或运行输出目录（模型、图表、csv 等）。
- `packages/`：本地 wheel 包目录（当无法联网时使用）。

（仓库可能包含额外脚本如 `rename.py`、`rename_files.py`、`merge/` 等）

## 环境与依赖

- 稳定运行的 Python 版本：3.12。
- 建议使用虚拟环境（venv 或 conda）。

### 安装依赖（联网）
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 离线安装（仓库包含 `packages/`）
```powershell
python -m pip install --no-index --find-links=packages -r requirements.txt
```

## 配置 `config.py`

在 `config.py` 中设置以下关键项：
```python
min_wavelength = 400
max_wavelength = 1100
step_wavelength = 1
data_type = "OV"  # 或 "IA"
flatten = False
add_noise = False
epochs = 50
delete_temp = True
```

根据你的光谱数据来源（Idea Optics / OceanView）调整 `data_type` 并确保 `min_wavelength`/`max_wavelength` 与 `step_wavelength` 匹配你的数据。flatten是专门用于单色光还原的选项，在还原非单色光时通常保持 False。add_noise是在还原时向数据添加噪声以增强鲁棒性的选项，可根据需要开启。epochs 是训练轮数，delete_temp 控制是否删除中间临时文件。

## 运行流程（快速开始）

1. 准备数据目录：
   - `rawdata/specs/` 放入原始光谱文件（文本）。
   - `rawdata/pics/` 放入滤光片图片（.bmp/.jpg）。
2. 设置 `config.py` 中的参数。
3. 训练：
   ```powershell
   python .\run.py
   ```
   运行时脚本会引导你用鼠标选择 ROI（滤光片区域）和背景区域。
4. 单张图片推理（示例）：

   在test文件夹中放置模型文件，输入归一化文件，以及待推理的图片

   修改 `test_image.py` 中的路径
   ```powershell
   python .\test_image.py
   ```
   输出文件会保存为 `test/model_output_<imagename>.csv`（脚本根据图像名自动命名），并弹出可视化图表。
5. 两份数据合并与可视化（示例）：

   在merge文件夹中放置待合并的csv文件和txt文件

   修改 `merge_image.py` 中的路径与文件名列表
   ```powershell
   python .\merge_image.py
   ```
   输出合并结果 CSV 和图表。

## 常见问题与排查

- pip 找不到：在 PowerShell 中请使用 `python -m pip`；如果你需要直接使用 `pip`，把 Python 安装目录下的 `Scripts` 添加到环境变量 PATH，或重新运行安装程序并勾选 "Add Python to PATH"。
- 合并后出现 NaN：若 `merge_image.py` 中合并结果为 NaN，可能原因是 CSV 读取后为 DataFrame 且索引不对齐，或原始数据包含 NaN/非数值。已在脚本中加入将数据转换为 numpy 数组并填充 NaN 的逻辑。
- 长度不匹配：若波长轴长度与输出向量长度不同，请确认 `config.step_wavelength` 与数据分辨率一致，或检查输入 CSV 的列数/表头是否被误读。

## 进阶配置与调整

- 如果希望使用双 y 轴或改变图例位置，可以在 `merge_image.py` 或 `test_image.py` 中自定义 matplotlib 参数。
- 如果不想用 NaN 填充为 0 的策略，可改为插值或前向填充（根据实验需求）。

## 以包（module）方式运行模块（推荐）

为了保证仓库中的导入（例如 `from src.system.config_manager import ...`）在本地与部署环境一致，推荐把 `src` 作为顶层包来运行 Python 模块。这样 Python 会正确识别 `src` 目录并找到包内模块。

1. 首先确保仓库根目录（包含 `src` 文件夹）为当前工作目录：

```powershell
cd D:\Code\Spectrum_ML
```

2. 以模块方式运行（示例：运行像 `pixelanalysis_utilities.py` 这样的模块）：

```powershell
# 以模块形式运行，Python 会把项目根加入搜索路径，从而允许 'from src.xxx import ...' 正常工作
python -m src.pixelanalysis.pixelanalysis_utilities
```

3. 如果你直接运行单个脚本（例如 `python src\pixelanalysis\pixelanalysis_utilities.py`），Python 会把脚本所在目录（`src/pixelanalysis`）当作模块根，导致 `from src...` 导入失败（报 "No module named 'src'"）。使用 `-m` 可以避免这个问题。

4. 另一种可选方式是设置环境变量 `PYTHONPATH`（临时）：

```powershell
$env:PYTHONPATH = 'D:\Code\Spectrum_ML'
python src\pixelanalysis\pixelanalysis_utilities.py
```

但推荐使用 `python -m` 的包运行方式，因为更稳定且符合 Python 的标准包导入机制。

## 在 VS Code 中避免生成临时运行文件（推荐设置）

如果你使用 VS Code 进行开发，某些扩展（如 Code Runner）在运行代码片段时可能会创建临时文件（例如 `tempCodeRunnerFile.py`）。为避免生成临时文件并保证包导入正常，推荐以下做法：

1. 使用官方 Python 扩展的“Run Python File in Terminal”（右键 -> Run Python File in Terminal）来运行当前文件；这不会创建临时文件并且在集成终端执行，行为与 `python <file>` 相同，但会保留当前工作目录为 workspace root（如果你希望以包方式运行，仍推荐 `python -m` 或使用 launch.json）。

2. 如果你仍想使用 Code Runner，可以在工作区设置中调整行为，已在本仓库添加了推荐的 workspace 设置：`.vscode/settings.json`，包含：

    - `code-runner.runInTerminal: true`（在集成终端运行）
    - `code-runner.saveFileBeforeRun: true`（运行前保存文件，避免使用临时副本）
    - `code-runner.cwd: ${workspaceFolder}`（将工作目录设为项目根，以便 `from src...` 导入正常工作）

    这些设置已经添加到仓库的 `.vscode/settings.json`，你可以直接使用，也可以根据个人偏好调整。

3. 更可靠的方式是使用 VS Code 的 launch 配置（`.vscode/launch.json`），并使用 `module` 选项或我之前添加的 prompt 型配置来以模块方式运行（`python -m <module>`）。按 F5 调试可以正确处理包导入并支持断点。

4. 如果你不想在工作区使用 Code Runner，可在扩展视图中将 Code Runner 禁用（Workspace 禁用），这样可以防止插件生成临时文件或改变运行方式。

如果你希望，我可以（任选其一）：
- 把 `pixelanalysis_utilities.py` 改为封装成 `main()` 并提供 `run_module.py` 快捷启动脚本；
- 或为你生成几个常用模块的一键调试配置（`.vscode/launch.json`）；
- 或把 Code Runner 在工作区禁用（通过编辑 workspace 的推荐扩展设置）。
