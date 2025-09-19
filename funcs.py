import csv

def calc_rois_gray_to_csv(rois_txt, src_dir, save_csv):
    import os
    import cv2
    # 读取ROI列表
    rois = []
    with open(rois_txt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                rois.append(tuple(map(int, parts)))
    # 获取所有jpg图片
    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    files.sort()
    # 写入csv
    with open(save_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for fname in files:
            img_path = os.path.join(src_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                row = ['error']*len(rois)
                writer.writerow(row)
                continue
            row = []
            for x, y, w, h in rois:
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    row.append('nan')
                else:
                    row.append(int(roi.mean()))
            writer.writerow(row)

import cv2

def get_files_in_directory(directory):
    import os
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def split(target, source, step):
    import pandas as pd
    df = pd.read_csv(source, header=None)
    # 按step间隔取列
    df_sampled = df.iloc[:, ::step]
    df_sampled.to_csv(target, index=False, header=False)

    

