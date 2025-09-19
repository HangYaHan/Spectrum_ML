import sys
import os
import pathlib
import pandas as pd

from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import colour

def calculate_CCT(spectrum, CCT, step=1, wavelength_s=400, wavelength_e=800):
    """
    Calculate the Correlated Color Temperature (CCT) from a given spectrum.

    Parameters:
    - spectrum: A list or array of spectral power distribution values.
    - target_file: Path to the file where the CCT result will be saved.
    - step: The wavelength step size (default is 1 nm).
    - wavelength_s: Starting wavelength (default is 400 nm).
    - wavelength_e: Ending wavelength (default is 700 nm).

    Returns:
    - cct: The calculated CCT value.
    """
    import colour
    import numpy as np

    # Generate wavelength array
    wavelengths = np.arange(wavelength_s, wavelength_e + step, step)

    # Ensure the spectrum length matches the wavelength array length
    if len(spectrum) != len(wavelengths):
        raise ValueError("Spectrum length does not match the wavelength range.")

    # Create a colour.SpectralDistribution object
    sd = colour.SpectralDistribution(dict(zip(wavelengths, spectrum)), name='Sample Spectrum')
    XYZ = colour.sd_to_XYZ(sd)
    xy = colour.XYZ_to_xy(XYZ)
    cct = colour.temperature.xy_to_CCT(xy)

    return int(cct)

def f_calculate_CCT(spectrum_csv, CCT_csv, step=1, wavelength_s=400, wavelength_e=800):
    spectrum_df = pd.read_csv(spectrum_csv)

    CCT_list = []
    for index, row in spectrum_df.iterrows():
        spectrum = row[0:].values  # Assume the first column is an identifier
        cct = calculate_CCT(spectrum, CCT_csv, step, wavelength_s, wavelength_e)
        CCT_list.append(cct)

    # Create DataFrame and save, overwrite file if it exists
    CCT_df = pd.DataFrame({'CCT': CCT_list})
    CCT_df.to_csv(CCT_csv, index=False)
    
def count_over_CCT(CCT_csv, threshold=8000):
    CCT_df = pd.read_csv(CCT_csv)
    count = (CCT_df['CCT'] > threshold).sum()
    return count

def rename_and_match_files(pic_dir, spec_dir):
    def rename_files_in_dir(dir):
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        files_with_mtime = [(f, os.path.getmtime(os.path.join(dir, f))) for f in files]
        files_with_mtime.sort(key=lambda x: x[1])
        num_files = len(files)
        width = len(str(num_files))
        for idx, (filename, _) in enumerate(files_with_mtime, 1):
            ext = pathlib.Path(filename).suffix
            new_name = f"{str(idx).zfill(width)}{ext}"
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
    

def filter_wavelength_range(spec_dir, min_wavelength=400, max_wavelength=800):
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

def spectrum_to_csv(spec_dir, target_dir):
    import csv
    save_csv = os.path.join(target_dir, 'spectrum.csv')
    files = [f for f in os.listdir(spec_dir) if f.lower().endswith('.txt')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    spectra = []
    for idx, fname in enumerate(files, 1):
        path = os.path.join(spec_dir, fname)
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        values.append(int(float(parts[1])))
                    except Exception:
                        values.append('nan')
        values = values[:401]
        if len(values) < 401:
            values += ['nan'] * (401 - len(values))
        spectra.append([idx] + values)
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in spectra:
            writer.writerow(row)

def extract_roi_from_image(pic_dir, target_dir):
    import cv2
    img_path = os.path.join(pic_dir, '0001.jpg')
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return

    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    rois = cv2.selectROIs("Select ROIs (ESC to finish)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    save_path = os.path.join(target_dir, 'rois.txt')
    with open(save_path, 'w') as f:
        for roi in rois:
            x, y, w, h = roi
            f.write(f"{x},{y},{w},{h}\n")

def calculate_roi_gray_values(pic_dir, target_dir):
    import os
    import cv2
    import csv
    rois_txt = os.path.join(target_dir, 'rois.txt')
    save_csv = os.path.join(target_dir, 'grey.csv')
    # 读取ROI列表
    rois = []
    with open(rois_txt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                rois.append(tuple(map(int, parts)))
    # 获取所有jpg图片
    files = [f for f in os.listdir(pic_dir) if f.lower().endswith('.jpg')]
    files.sort()
    # 写入csv，首列为序号
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, fname in enumerate(files, 1):
            img_path = os.path.join(pic_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                row = [idx] + ['error']*len(rois)
                writer.writerow(row)
                continue
            row = [idx]
            for x, y, w, h in rois:
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    row.append('nan')
                else:
                    row.append(int(roi.mean()))
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

def calculate_cct_for_spectra(spectrum_csv_path, cct_csv_path, step=1, wavelength_s=400, wavelength_e=800):
    pass

def count_cct_above_threshold(cct_csv_path, threshold=8000):
    pass
    

if __name__ == "__main__":

    # Here is the test environment. For usage, please refer to the dataprocess.py file.

    config.print_csv_shape(os.path.join(config.grey_csv_folder, "grey.csv"))
    config.print_csv_shape(os.path.join(config.spec_csv_folder, "spectrum.csv"))