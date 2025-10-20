# Spectrum_ML

Remember to adjust min_wavelength and max_wavelength parameters in config.

## Quick: run camera screenshot script (Windows PowerShell)

Run `test_camera.py` to open camera index 1. Press Enter to save a timestamped screenshot into the `screenshots` folder, or ESC to exit.

Example:

	python .\test_camera.py

If your camera is not at index 1, change the index in the script (VideoCapture(1) -> VideoCapture(0) or other).