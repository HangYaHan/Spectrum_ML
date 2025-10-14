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

    train_utilities.train_resnet18_spectrum(
        min_wavelength=config.min_wavelength,
        max_wavelength=config.max_wavelength,
        step=config.step_wavelength,
        src_csv=os.path.join(target_dir_timestamped, "grey.csv"),
        target_csv=os.path.join(target_dir_timestamped, "spectrum.csv"),
        save_path=target_dir_timestamped,
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device="cpu",
        random_seed=42,
        optimizer="adam",
        loss_fn="mse"
    )