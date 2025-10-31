import pandas as pd
import os

# -------------------------------
# System Parameters
# -------------------------------

source_folder = ".\\rawdata"
temp_folder = ".\\temp"
target_folder = ".\\result\\result"
test_dir = ".\\test\\"
merge_dir = ".\\merge\\"

# -------------------------------
# Data Process Parameters
# -------------------------------

min_wavelength = 400
max_wavelength = 1100
step_wavelength = 1
delete_temp = False

# data_type = "IA" # Idea Optics
data_type = "OV" # Ocean View

# -------------------------------
# Machine Learning Parameters
# -------------------------------

epochs = 2000

# -------------------------------
# Test Parameters
# -------------------------------

flatten = False
add_noise = False
offset = 0

if __name__ == "__main__":
    print("This is a configuration file and it stores some global variables.")

