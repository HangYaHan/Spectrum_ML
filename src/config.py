import pandas as pd
import os

original_pics_folder = ".\\RawData\\pics"
original_specs_folder = ".\\RawData\\specs"

grey_csv_folder = ".\\ProcessedData\\FinalCSV"
spec_csv_folder = ".\\ProcessedData\\FinalCSV"

def print_csv_shape(csv_filename):
    df = pd.read_csv(csv_filename)
    print(f"Shape of '{csv_filename}': {df.shape}")

if __name__ == "__main__":
    print("This is a configuration file and it can run some basic tests.")
    print_csv_shape(grey_csv_folder + "\\grey.csv")
    print_csv_shape(spec_csv_folder + "\\spectrum.csv")
