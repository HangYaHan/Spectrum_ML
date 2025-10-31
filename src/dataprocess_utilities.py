import csv
import sys
import os
import pathlib
import pandas as pd
import shutil

from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

def rename_and_match_files(pic_dir, spec_dir):
    def rename_files_in_dir(dir):
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        files_with_mtime = [(f, os.path.getmtime(os.path.join(dir, f))) for f in files]
        files_with_mtime.sort(key=lambda x: x[1])
        num_files = len(files)
        width = len(str(num_files))
        for idx, (filename, _) in enumerate(files_with_mtime, 1):
            ext = pathlib.Path(filename).suffix
            new_name = f"{str(idx).zfill(width)}888{ext}"
            src = os.path.join(dir, filename)
            dst = os.path.join(dir, new_name)
            os.rename(src, dst)
        return num_files

    num_pic = rename_files_in_dir(pic_dir)
    num_spec = rename_files_in_dir(spec_dir)

    if num_pic != num_spec:
        raise ValueError(f"Number of pictures ({num_pic}) does not match number of spectra ({num_spec}).")
    else:
        pass

def remove_header_lines(spec_dir, num_lines=14):
    def remove_lines_from_file(file_path, num_lines):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        with open(file_path, 'w') as file:
            file.writelines(lines[num_lines:])
    for filename in os.listdir(spec_dir):
        if filename.endswith(".txt"):
            remove_lines_from_file(os.path.join(spec_dir, filename), num_lines)

def convert_spectrum_to_int(spec_dir):
    def convert_to_int_file(file_path):
        groups = defaultdict(list)
        # Read and group data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    key = int(float(parts[0]))
                    value = float(parts[1])
                    groups[key].append(value)
                except Exception:
                    continue
        # Calculate average and overwrite file
        with open(file_path, 'w', encoding='utf-8') as f:
            for key in sorted(groups.keys()):
                avg = int(sum(groups[key]) / len(groups[key]))
                f.write(f"{key}\t{avg}\n")

    for filename in os.listdir(spec_dir):
        if filename.endswith(".txt"):
            convert_to_int_file(os.path.join(spec_dir, filename))
    

def filter_wavelength_range(spec_dir, min_wavelength, max_wavelength):
    for filename in os.listdir(spec_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(spec_dir, filename)
            lines_to_keep = []
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    try:
                        value = float(parts[0])
                    except Exception:
                        continue
                    if min_wavelength <= value <= max_wavelength:
                        lines_to_keep.append(line)
            with open(file_path, 'w', encoding='utf-8') as fout:
                fout.writelines(lines_to_keep)

def spectrum_to_csv(spec_dir, target_dir, min_wavelength, max_wavelength):
    import csv
    save_csv = os.path.join(target_dir, 'spectrum.csv')
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    files = [f for f in os.listdir(spec_dir) if f.lower().endswith('.txt')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    spectra = []
    for idx, fname in enumerate(files, 1):
        path = os.path.join(spec_dir, fname)
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # print(f"Raw line: {line.strip()}, Parsed parts: {parts}")
                if len(parts) >= 2:
                    try:
                        value = int(float(parts[1]))
                        values.append(value)
                    except Exception as e:
                        print(f"Error parsing value: {parts[1]}, Error: {e}")
                        values.append('nan')
                else:
                    print(f"Line skipped due to format: {line.strip()}")
        values = values[:max_wavelength - min_wavelength + 1]
        if len(values) < max_wavelength - min_wavelength + 1:
            values += ['nan'] * ((max_wavelength - min_wavelength + 1) - len(values))
        spectra.append([idx] + values)
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in spectra:
            writer.writerow(row)

def extract_roi_from_image(pic_dir, target_dir):
    import cv2
    import os

    # 检查 target_dir 中是否已存在 rois.txt
    rois_path = os.path.join(target_dir, 'rois.txt')
    if os.path.exists(rois_path):
        print(f"rois.txt already exists in {target_dir}, skipping ROI extraction.")
        return

    # 指定圖片文件名
    img_filename = '101888.bmp'
    img_path = os.path.join(pic_dir, img_filename)
    if not os.path.exists(img_path):
        print(f"Specified image file not found: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    zoom_factor = 1
    img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    rois = cv2.selectROIs("Select ROIs (ESC to finish)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    save_path = os.path.join(target_dir, 'rois.txt')
    with open(save_path, 'w') as f:
        for roi in rois:
            x, y, w, h = roi
            # rescale back to original size if zoomed
            x, y, w, h = int(x/zoom_factor), int(y/zoom_factor), int(w/zoom_factor), int(h/zoom_factor)
            f.write(f"{x},{y},{w},{h}\n")

def calculate_roi_gray_values(pic_dir, target_dir):
    import os
    import cv2
    import csv

    # 读取 bgrois.txt 中的 x 值
    bgrois_path = os.path.join(target_dir, 'bgrois.txt')
    if not os.path.exists(bgrois_path):
        print(f"bgrois.txt not found in {target_dir}.")
        return

    with open(bgrois_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split(',')
        if len(parts) < 5:
            print(f"Invalid format in bgrois.txt: {line}")
            return
        try:
            x_value = float(parts[4])
        except ValueError:
            print(f"Invalid x value in bgrois.txt: {parts[4]}")
            return

    if x_value == 0:
        print("x value in bgrois.txt is zero, cannot divide by zero.")
        return

    rois_txt = os.path.join(target_dir, 'rois.txt')
    save_csv = os.path.join(target_dir, 'grey.csv')

    # 读取 rois
    rois = []
    with open(rois_txt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                rois.append(tuple(map(int, parts)))

    # 读取图片并计算灰度值
    files = [f for f in os.listdir(pic_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    files.sort()

    # 写入 csv，第一列是索引
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, fname in enumerate(files, 1):
            img_path = os.path.join(pic_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                row = [idx] + ['error'] * len(rois)
                writer.writerow(row)
                continue
            row = [idx]
            for x, y, w, h in rois:
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    row.append('nan')
                else:
                    avg_gray = roi.mean() / x_value  # 除以 x 值
                    row.append(avg_gray)
            writer.writerow(row)

def load_and_check_csv(target_dir):
    import os
    import pandas as pd
    spectrum_path = os.path.join(target_dir, 'spectrum.csv')
    grey_path = os.path.join(target_dir, 'grey.csv')
    spectrum_df = pd.read_csv(spectrum_path, header=None)
    grey_df = pd.read_csv(grey_path, header=None)
    print(f"spectrum.csv shape: {spectrum_df.shape}")
    print(f"grey.csv shape: {grey_df.shape}")
    if spectrum_df.shape[0] == grey_df.shape[0]:
        print("Row count is equal.")
    else:
        print(f"Row count is NOT equal! spectrum.csv: {spectrum_df.shape[0]}, grey.csv: {grey_df.shape[0]}")

def copy_original_data(temp_dir, src_dir):
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    shutil.copytree(src_dir, temp_dir)

def if_nan_in_csv(csv_path):
        df = pd.read_csv(csv_path, header=None)
        has_nan = df.isna().any().any()
        has_error = (df == 'error').any().any()
        print(f"{csv_path} contains nan: {has_nan}, contains 'error': {has_error}")
        return has_nan or has_error

def cleanup_temp_files(temp_dir):
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def remove_head_and_get_intensity_file(source_file):
    if not os.path.isfile(source_file):
        print(f"File not found: {source_file}")
        return None

    with open(source_file, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过header行
        data = [int(float(row[1])) for row in reader if len(row) > 1]  # 只保留第二列并取整

    with open(source_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for value in data:
            writer.writerow([value])  # 保存第二列数据覆盖原文件

    # print(f"Processed file saved: {source_file}")
    return source_file

def remove_head_and_get_intensity(spec_dir):
    for filename in os.listdir(spec_dir):
        file_path = os.path.join(spec_dir, filename)
        if os.path.isfile(file_path) and filename.endswith('.csv'):
            remove_head_and_get_intensity_file(file_path)
    
def merge_csv_to_one(spec_dir, target_dir):
    import csv
    from operator import itemgetter

    # 获取所有 CSV 文件，并按创建时间排序
    files = [f for f in os.listdir(spec_dir) if f.lower().endswith('.csv')]
    files_with_time = [(f, os.path.getctime(os.path.join(spec_dir, f))) for f in files]
    files_with_time.sort(key=itemgetter(1))

    # 合并数据
    merged_data = []
    for file, _ in files_with_time:
        file_path = os.path.join(spec_dir, file)
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            row_data = [row[0] for row in reader]  # 读取单列数据
            merged_data.append(row_data)  # 每个文件的数据作为一行

    # 保存到目标文件
    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, 'spectrum.csv')
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(merged_data)  # 按行保存

    print(f"Merged CSV saved to: {output_file}")

if __name__ == "__main__":
    test_file = "temp/specs/001.csv"  # 替换为你的测试文件路径
    header = remove_head_and_get_intensity_file(test_file)
    if header:
        print(f"Header of the file: {header}")


def check_and_copy_rois(temp_dir, target_dir):
    """
    Check if rois.txt and bgrois.txt exist in temp_dir and copy them to target_dir if found.
    """
    files_to_check = ['rois.txt', 'bgrois.txt']
    copied_files = []

    for file_name in files_to_check:
        file_path = os.path.join(temp_dir, file_name)
        if os.path.exists(file_path):
            print(f"Found existing {file_name} in {temp_dir}, copying to {target_dir}.")
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(file_path, os.path.join(target_dir, file_name))
            copied_files.append(file_name)

    if copied_files:
        print(f"Copied files: {', '.join(copied_files)}")
        return True

    print("No files were copied.")
    return False

def extract_bgroi_from_image(pic_dir, target_dir):
    import cv2
    import os

    # 检查 target_dir 中是否已存在 bgrois.txt
    bgrois_path = os.path.join(target_dir, 'bgrois.txt')
    if os.path.exists(bgrois_path):
        print(f"bgrois.txt already exists in {target_dir}, skipping ROI extraction.")
        return

    # 指定图片文件名
    img_filename = '101888.bmp'
    img_path = os.path.join(pic_dir, img_filename)
    if not os.path.exists(img_path):
        print(f"Specified image file not found: {img_path}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    # 让用户选择 ROI
    roi = cv2.selectROI("Select ROI (ESC to finish)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No ROI selected.")
        return

    x, y, w, h = roi
    selected_roi = img[y:y+h, x:x+w]

    # 计算 ROI 的平均灰度值
    if selected_roi.size == 0:
        print("Selected ROI is empty.")
        return

    avg_gray_value = selected_roi.mean()

    # 保存结果到 bgrois.txt
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, 'bgrois.txt')
    with open(save_path, 'w') as f:
        f.write(f"{x},{y},{w},{h},{avg_gray_value:.2f}\n")

    print(f"ROI average gray value saved to: {save_path}")