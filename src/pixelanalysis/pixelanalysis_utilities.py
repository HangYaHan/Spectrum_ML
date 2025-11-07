from src.system.config_manager import get, set
from pathlib import Path
import shutil
import os
from typing import List, Tuple
import numpy as np
from datetime import datetime

source_folder = get("system_parameters.source_folder")
temp_folder = get("system_parameters.temp_folder")


def create_grayscale_image(gray_value, dir_path, filename, size=(100, 100), fmt=None):
    """Create and save a solid grayscale image file.

    Parameters:
      gray_value: int 0-255 or float 0.0-1.0 (scaled to 0-255)
      dir_path:   output directory (str or Path). Created if missing.
      filename:   output file name (may include extension). If missing extension,
                  `fmt` or default PNG is used to decide extension.
      size:       (width, height) tuple
      fmt:        optional format hint ('PNG','JPEG','BMP')

    Returns:
      pathlib.Path to the saved image.
    """

    # Normalize gray value to integer 0-255
    if isinstance(gray_value, float):
        if not (0.0 <= gray_value <= 1.0):
            raise ValueError("float gray_value must be between 0.0 and 1.0")
        gray_int = int(round(gray_value * 255))
    elif isinstance(gray_value, int):
        if not (0 <= gray_value <= 255):
            raise ValueError("int gray_value must be between 0 and 255")
        gray_int = gray_value
    else:
        raise TypeError("gray_value must be int (0-255) or float (0.0-1.0)")

    # Prepare output directory
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    out_path = dir_path / filename

    # Determine output format/extension
    if out_path.suffix == "":
        chosen_fmt = (fmt or "PNG").upper()
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg", "BMP": ".bmp"}
        out_path = out_path.with_suffix(ext_map.get(chosen_fmt, ".png"))
    else:
        s = out_path.suffix.lower()
        if s in (".png",):
            chosen_fmt = "PNG"
        elif s in (".jpg", ".jpeg"):
            chosen_fmt = "JPEG"
        elif s in (".bmp",):
            chosen_fmt = "BMP"
        else:
            chosen_fmt = (fmt or "PNG").upper()

    # Create image using Pillow
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required to create images. Install with: python -m pip install pillow")

    img = Image.new("L", size, color=gray_int)
    save_kwargs = {}
    if chosen_fmt == "JPEG":
        save_kwargs["quality"] = 95

    img.save(out_path, format=chosen_fmt, **save_kwargs)
    return out_path


def copy_pics_to_temp(source_folder: str, temp_folder: str, pics_subdir_name: str = "pics") -> Path:
    """
    Copy the `pics` subdirectory from `source_folder` into a timestamped folder under `temp_folder`.

    Returns the path to the copied directory inside temp_folder.
    This function does not modify the original data.
    """
    src = Path(source_folder) / pics_subdir_name
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source pics directory not found: {src}")

    dst_parent = Path(temp_folder)
    dst_parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_parent / f"pics_copy_{ts}"
    shutil.copytree(src, dst)
    return dst


def _get_image_file_list(folder: Path, exts=None) -> List[Path]:
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def load_images_as_grayscale_array(folder: Path) -> Tuple[List[Path], np.ndarray]:
    """
    Load all supported images from `folder` as a 3D numpy array with shape
    (n_images, height, width). Images are converted to grayscale.

    Returns (paths_list, array)
    """
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required to load images. Install with: python -m pip install pillow")

    files = _get_image_file_list(folder)
    if not files:
        raise FileNotFoundError(f"No supported image files found in {folder}")

    imgs = []
    for p in files:
        with Image.open(p) as im:
            gray = im.convert("L")
            arr = np.array(gray, dtype=np.float32)
            imgs.append(arr)

    stack = np.stack(imgs, axis=0)  # shape (n, h, w)
    return files, stack


def compute_cumulative_and_average_deltas(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a stack of grayscale images with shape (n, h, w), compute per-pixel
    cumulative delta (sum of absolute differences between consecutive frames)
    and average delta (cumulative / (n-1)).
    """
    if stack.ndim != 3:
        raise ValueError("stack must have shape (n, h, w)")
    n = stack.shape[0]
    if n < 2:
        cum = np.zeros(stack.shape[1:], dtype=np.float32)
        avg = cum.copy()
        return cum, avg

    diffs = np.abs(np.diff(stack, axis=0))
    cum = np.sum(diffs, axis=0)
    avg = cum / float(n - 1)
    return cum.astype(np.float32), avg.astype(np.float32)


def save_matrix_csv(mat: np.ndarray, out_path: Path):
    """Save 2D numpy array to CSV using comma separator."""
    np.savetxt(out_path, mat, delimiter=",", fmt="%.6f")


def overlay_highlights_three(base_image_path: Path, cum: np.ndarray, avg: np.ndarray, out_dir: Path, top_pct: float = 0.3, alpha: float = 0.6) -> Tuple[Path, Path, Path]:
    """
    Create three visualization images under out_dir based on cum/avg masks:
      - cumulative-only highlights (red) -> saved as 'highlight_cumulative.png'
      - average-only highlights (blue) -> saved as 'highlight_average.png'
      - combined highlights (red=cum-only, blue=avg-only, green=both) -> saved as 'highlight_combined.png'

    Returns tuple(paths): (cum_path, avg_path, combined_path)
    """
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow is required to create overlay images. Install with: python -m pip install pillow")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Image.open(base_image_path).convert("L")
    base_arr = np.array(base, dtype=np.float32)

    h, w = base_arr.shape
    if cum.shape != (h, w) or avg.shape != (h, w):
        raise ValueError("cum/avg shapes must match base image shape")

    # adaptive thresholding: ensure at least a small number of pixels selected
    size = cum.size
    min_pixels = max(1, int(0.001 * size))  # at least 0.1% of pixels or 1 pixel

    def adaptive_mask(arr, init_top_pct):
        pct = float(init_top_pct)
        # clamp
        pct = min(max(pct, 0.0001), 0.99)
        while True:
            thresh = np.quantile(arr, 1.0 - pct)
            mask = arr >= thresh
            count = int(np.count_nonzero(mask))
            if count >= min_pixels or pct >= 0.99:
                return mask
            # relax threshold to include more pixels
            pct = min(0.99, pct + max(0.05, pct * 0.5))

    mask_cum = adaptive_mask(cum, top_pct)
    mask_avg = adaptive_mask(avg, top_pct)
    mask_both = mask_cum & mask_avg

    # Helper to composite a colored overlay image from a mask
    def make_overlay(mask, color):
        # start from base gray as RGB
        rgb = np.stack([base_arr, base_arr, base_arr], axis=-1).astype(np.float32)
        inv_alpha = 1.0 - alpha
        for c in range(3):
            rgb[..., c] = (rgb[..., c] * inv_alpha + (mask.astype(np.float32) * color[c]) * alpha)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # cumulative image: include both cum-only and overlap (so user sees full cum mask)
    cum_rgb = make_overlay(mask_cum, (255, 0, 0))
    # average image: include both avg-only and overlap
    avg_rgb = make_overlay(mask_avg, (0, 0, 255))
    # combined: red for cum-only, blue for avg-only, green for both
    combined_rgb = np.stack([base_arr, base_arr, base_arr], axis=-1).astype(np.float32)
    inv_alpha = 1.0 - alpha
    # apply cum-only red
    for c in range(3):
        combined_rgb[..., c] = combined_rgb[..., c] * inv_alpha
    # add colors
    # cum-only
    combined_rgb[..., 0] += (mask_cum & ~mask_both).astype(np.float32) * 255 * alpha
    # avg-only
    combined_rgb[..., 2] += (mask_avg & ~mask_both).astype(np.float32) * 255 * alpha
    # both -> green
    combined_rgb[..., 1] += (mask_both).astype(np.float32) * 255 * alpha
    combined_arr = np.clip(combined_rgb, 0, 255).astype(np.uint8)

    # save three images
    cum_path = out_dir / "highlight_cumulative.png"
    avg_path = out_dir / "highlight_average.png"
    combined_path = out_dir / "highlight_combined.png"

    Image.fromarray(cum_rgb).save(cum_path)
    Image.fromarray(avg_rgb).save(avg_path)
    Image.fromarray(combined_arr).save(combined_path)

    return cum_path, avg_path, combined_path


if __name__ == "__main__":
    # example usage (no-op when running as module)
    pass