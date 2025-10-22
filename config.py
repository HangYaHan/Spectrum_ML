import pandas as pd
import os

source_folder = ".\\rawdata"
temp_folder = ".\\temp"
target_folder = ".\\result\\result"

test_dir = ".\\test\\"

min_wavelength = 800
max_wavelength = 1050
step_wavelength = 1

data_type = "IA" # Idea Optics
# data_type = "OV" # Ocean View

epochs = 2000

delete_temp = False

if __name__ == "__main__":
    print("This is a configuration file and it stores some global variables.")

