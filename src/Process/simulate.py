import pandas as pd
import os
import sys

from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

def amplify_grey(target_csv, source_csv, amplify_factor=1.6):
	# 读取原始灰度 CSV
	df = pd.read_csv(source_csv, header=None)
	# 只放大除第一列外的灰度值，第一列序号保持不变
	def amplify_cell(x):
		try:
			val = float(x)
			val = int(val * amplify_factor)
			return min(val, 255)
		except Exception:
			return x  # 保留非数值内容，如 'nan' 或 'error'
	# 第一列不变，其余列放大
	df.iloc[:, 1:] = df.iloc[:, 1:].applymap(amplify_cell)
	# 保存到目标 CSV
	df.to_csv(target_csv, header=False, index=False)

if __name__ == "__main__":
    amplify_grey(os.path.join(config.final_folder, "grey_amplified.csv"),
                 os.path.join(config.final_folder, "grey.csv"),
                 amplify_factor=1.6)
