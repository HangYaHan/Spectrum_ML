from pathlib import Path
from src.pixelanalysis.pixelanalysis_utilities import (
	copy_pics_to_temp,
	load_images_as_grayscale_array,
	compute_cumulative_and_average_deltas,
	save_matrix_csv,
	overlay_highlights_three,
)
import numpy as np


def process_pics(source_folder: str, temp_folder: str, pics_subdir: str = "pics", top_pct: float = 0.3):
	"""
	Full pipeline:
	  - copy pics from source -> temp (safe, no modification)
	  - load grayscale images
	  - compute cumulative and average per-pixel deltas
	  - save CSVs (cumulative.csv, average.csv) into copied folder
	  - create visualization by opening the last-created file in source/pics and
		overlaying highlights, saved as highlighted.png in copied folder

	Returns a dict with paths to outputs.
	"""
	src = Path(source_folder)
	# copy pics
	dst = copy_pics_to_temp(source_folder, temp_folder, pics_subdir)

	# load images from copied folder
	files, stack = load_images_as_grayscale_array(dst)

	cum, avg = compute_cumulative_and_average_deltas(stack)

	# save CSVs
	cum_path = dst / "cumulative.csv"
	avg_path = dst / "average.csv"
	save_matrix_csv(cum, cum_path)
	save_matrix_csv(avg, avg_path)

	# find last-created file in original source/pics for visualization base
	src_pics = Path(source_folder) / pics_subdir
	candidates = [p for p in src_pics.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
	if not candidates:
		raise FileNotFoundError(f"No image files found in source pics: {src_pics}")
	last = max(candidates, key=lambda p: p.stat().st_ctime)

	# create three visualization images and return their paths
	cum_img, avg_img, combined_img = overlay_highlights_three(last, cum, avg, dst, top_pct=top_pct)

	return {
		"copied_folder": dst,
		"cumulative_csv": cum_path,
		"average_csv": avg_path,
		"highlight_cumulative": cum_img,
		"highlight_average": avg_img,
		"highlight_combined": combined_img,
	}


if __name__ == "__main__":
	# Example invocation using configured folders in config_manager
	try:
		from src.system.config_manager import get
		sf = get("system_parameters.source_folder")
		tf = get("system_parameters.temp_folder")
		out = process_pics(sf, tf)
		print("Pipeline outputs:")
		for k, v in out.items():
			print(k, v)
	except Exception as e:
		print("Error running pipeline:", e)

