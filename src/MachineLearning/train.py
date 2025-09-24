import os
import sys
import torchvision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MachineLearning.train_utilities_Spectrum import train_resnet18_spectrum
import config

if __name__ == "__main__":
    train_resnet18_spectrum(
        min_wavelength=800,
        max_wavelength=1050,
        step=1,
        src_csv=os.path.join(config.final_folder, "grey.csv"),
        target_csv=os.path.join(config.final_folder, "spectrum.csv"),
        save_path=os.path.join(config.final_folder, "resnet18_spectrum"),
        epochs=1500,
        batch_size=8,
        learning_rate=5e-2,
        device="cpu",
        random_seed=42,
        optimizer="adam",
        loss_fn="mse"
    )