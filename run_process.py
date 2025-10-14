import config
from src import dataprocess_pipeline
from src import train_utilities

import os
from datetime import datetime

if __name__ == "__main__":
    print("This is the complete pipeline entry point of the project.")

    # Append current timestamp to target_dir
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    target_dir_timestamped = f"{config.target_folder}_{timestamp}"


    dataprocess_pipeline.data_process_pipeline(
        target_dir=target_dir_timestamped,
        src_dir=config.source_folder,
        temp_dir=config.temp_folder,
        min_wavelength=config.min_wavelength,
        max_wavelength=config.max_wavelength
    )