import pandas as pd
import os

source_folder = ".\\rawdata"
temp_folder = ".\\temp"
target_folder = ".\\result\\result"

test_dir = ".\\test\\"

min_wavelength = 800
max_wavelength = 1050
step_wavelength = 1

epochs = 2000

def print_csv_shape(csv_filename):
    df = pd.read_csv(csv_filename)
    print(f"Shape of '{csv_filename}': {df.shape}")

if __name__ == "__main__":
    print("This is a configuration file and it stores some global variables.")

