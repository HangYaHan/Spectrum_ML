import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

if __name__ == "__main__":
    config.print_csv_shape(os.path.join(config.grey_csv_folder, "grey.csv"))
    config.print_csv_shape(os.path.join(config.spec_csv_folder, "spectrum.csv"))