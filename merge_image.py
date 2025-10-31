import numpy as np
import matplotlib.pyplot as plt
import os
from src import test_utilities
from src import dataprocess_utilities
import config
import pandas as pd

if __name__ == "__main__":
    # source data paths
    model_out_1_path = config.merge_dir + "658.csv"
    model_out_2_path = config.merge_dir + "662.csv"
    OV_data_1_path = config.test_dir + "658.txt"
    OV_data_2_path = config.test_dir + "662.txt"

    model_out_1 = pd.read_csv(model_out_1_path)
    model_out_2 = pd.read_csv(model_out_2_path)

    # ---- Diagnostics & normalization ----
    # Convert possible single-column DataFrame/Series to 1D numpy float arrays
    # Use `coerce` to turn non-numeric values into NaN so we can detect/fill them.
    model_out_1 = model_out_1.squeeze()
    model_out_2 = model_out_2.squeeze()

    model_out_1 = pd.to_numeric(model_out_1, errors='coerce').to_numpy(dtype=float)
    model_out_2 = pd.to_numeric(model_out_2, errors='coerce').to_numpy(dtype=float)

    print(f"model_out_1 shape: {model_out_1.shape}, NaNs: {np.isnan(model_out_1).sum()}")
    print(f"model_out_2 shape: {model_out_2.shape}, NaNs: {np.isnan(model_out_2).sum()}")

    # If there are NaNs, fill them with 0 for the purpose of taking the maximum
    if np.isnan(model_out_1).any() or np.isnan(model_out_2).any():
        print("Warning: NaN values detected in model outputs — filling NaNs with 0 before merging.")
        model_out_1 = np.nan_to_num(model_out_1, nan=0.0, posinf=0.0, neginf=0.0)
        model_out_2 = np.nan_to_num(model_out_2, nan=0.0, posinf=0.0, neginf=0.0)
    # ---------------------------------------

    # Check if spectrum.csv exists in config.merge_dir
    spectrum_csv_path = os.path.join(config.merge_dir, "spectrum.csv")
    if not os.path.exists(spectrum_csv_path):
        dataprocess_utilities.remove_header_lines(config.merge_dir, num_lines=14)
        dataprocess_utilities.convert_spectrum_to_int(config.merge_dir)
        dataprocess_utilities.filter_wavelength_range(config.merge_dir, config.min_wavelength, config.max_wavelength)
        dataprocess_utilities.spectrum_to_csv(config.merge_dir, config.merge_dir, config.min_wavelength, config.max_wavelength)

    # Read spectrum.csv, which is expected to have exactly two rows
    spectrum_data = pd.read_csv(spectrum_csv_path, header=None).values

    if spectrum_data.shape[0] != 2:
        raise ValueError(f"Expected spectrum.csv to have exactly two rows, but found {spectrum_data.shape[0]} rows.")

    # Remove the first two columns from spectrum_data
    spectrum_data = spectrum_data[:, 2:]

    # Assign the remaining data to OV_data_1 and OV_data_2
    OV_data_1, OV_data_2 = spectrum_data

    # Ensure OV_data_1 and OV_data_2 are 1D arrays
    OV_data_1 = OV_data_1.flatten()
    OV_data_2 = OV_data_2.flatten()

    # Regenerate sample_x to match the length of the outputs
    start_wl = float(config.min_wavelength)
    length = len(model_out_1)
    sample_x = start_wl + np.arange(length) * float(config.step_wavelength)

    # Plot model_out_1, model_out_2, OV_data_1, and OV_data_2 on the same figure
    plt.figure(figsize=(10, 6))
    plt.plot(sample_x, model_out_1, label=f'Model Output 1 ({os.path.basename(model_out_1_path)})', color='blue')
    plt.plot(sample_x, model_out_2, label=f'Model Output 2 ({os.path.basename(model_out_2_path)})', color='red')
    plt.plot(sample_x, OV_data_1, label=f'OV Data 1 ({os.path.basename(OV_data_1_path)})', color='orange')
    plt.plot(sample_x, OV_data_2, label=f'OV Data 2 ({os.path.basename(OV_data_2_path)})', color='purple')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Model Outputs and OV Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Merge model outputs and OV data by MAX
    merged_model_out = np.maximum(model_out_1, model_out_2)
    merged_OV_data = np.maximum(OV_data_1, OV_data_2)

    # Print a small sample for debugging
    print("Merged Model Output sample (first 20):", merged_model_out[:20])
    print(f"Merged Model Output shape: {merged_model_out.shape}, NaNs: {np.isnan(merged_model_out).sum()}")

    # Plot merged_model_out and merged_OV_data on the same figure
    # Ensure lengths match; if not, truncate to the shortest for plotting
    if len(sample_x) != len(merged_model_out) or len(sample_x) != len(merged_OV_data):
        minlen = min(len(sample_x), len(merged_model_out), len(merged_OV_data))
        print(f"Warning: length mismatch among sample_x/merged arrays — truncating to {minlen} points for plotting")
        sx = sample_x[:minlen]
        mmo = merged_model_out[:minlen]
        mov = merged_OV_data[:minlen]
    else:
        sx = sample_x
        mmo = merged_model_out
        mov = merged_OV_data

    plt.figure(figsize=(10, 6))
    plt.plot(sx, mmo, label=f'Merged Model Output ({os.path.basename(model_out_1_path)} + {os.path.basename(model_out_2_path)})', color='green', linewidth=1.5)
    plt.plot(sx, mov, label=f'Merged OV Data ({os.path.basename(OV_data_1_path)} + {os.path.basename(OV_data_2_path)})', color='brown', linestyle='--', linewidth=1.5)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Merged Model Output and Merged OV Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sx, mmo, label=f'Merged Model Output ({os.path.basename(model_out_1_path)} + {os.path.basename(model_out_2_path)})', color='green', linewidth=1.5)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Merged Model Output and Merged OV Data')
    plt.legend()
    plt.grid(True)
    plt.show()

