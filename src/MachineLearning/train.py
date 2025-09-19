import os
import sys
import torchvision

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MachineLearning.train_utilities_Spectrum import train_resnet18_spectrum
import config

if __name__ == "__main__":
    train_resnet18_spectrum(
        src_csv=os.path.join(config.final_folder, "spectrum.csv"),
        target_csv=os.path.join(config.final_folder, "CCT.csv"),
        save_path=os.path.join(config.final_folder, "resnet18_spectrum.pth"),
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device="cpu",
        random_seed=42,
        optimizer="adam",
        loss_fn="mse"
    )