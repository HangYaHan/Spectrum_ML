import sys
import os
import pathlib
import pandas as pd

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
    return num_pic, num_spec

def remove_header_lines(spec_path, spec_clean_path, num_lines=14):
    

if __name__ == "__main__":

    # Here is the test environment. For usage, please refer to the dataprocess.py file.

    config.print_csv_shape(os.path.join(config.grey_csv_folder, "grey.csv"))
    config.print_csv_shape(os.path.join(config.spec_csv_folder, "spectrum.csv"))