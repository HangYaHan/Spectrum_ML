import pandas as pd
import os

original_pics_folder = ".\\RawData\\pics"
original_specs_folder = ".\\RawData\\specs"

final_folder = ".\\ProcessedData\\FinalCSV"

def print_csv_shape(csv_filename):
    df = pd.read_csv(csv_filename)
    print(f"Shape of '{csv_filename}': {df.shape}")

if __name__ == "__main__":
    print("This is a configuration file and it can run some basic tests.")

