import os
from . import dataprocess_utilities

def data_process_pipeline(target_dir, src_dir, temp_dir, min_wavelength=400, max_wavelength=800):
    """
    Main pipeline for data preprocessing and feature extraction.
    Each step should be implemented as a separate function.
    """

    # 0. Make a copy of original data to temp folder
    dataprocess_utilities.copy_original_data(temp_dir, src_dir)
    spec_dir = os.path.join(temp_dir, "specs")
    pic_dir = os.path.join(temp_dir, "pics")
    # Create target_dir if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 1. Rename and match files
    dataprocess_utilities.rename_and_match_files(pic_dir, spec_dir)
    print("Files renamed and matched.")

    # 2. Remove header lines from spectrum files
    dataprocess_utilities.remove_header_lines(spec_dir, num_lines=14)
    print("Header lines removed from spectrum files.")

    # 3. Convert spectrum data to integer
    dataprocess_utilities.convert_spectrum_to_int(spec_dir)
    print("Spectrum data converted to integer.")

    # 4. Filter spectrum to target wavelength range
    dataprocess_utilities.filter_wavelength_range(spec_dir, min_wavelength, max_wavelength)
    print(f"Spectrum data filtered to target wavelength range: {min_wavelength} - {max_wavelength} nm.")

    # 5. Convert spectrum to CSV
    dataprocess_utilities.spectrum_to_csv(spec_dir, target_dir, min_wavelength, max_wavelength)
    print("Spectrum data converted to CSV.")

    # --- Spectrum process complete ---
    print("--- Spectrum processing complete ---")

    # 6. Extract ROIs from images
    dataprocess_utilities.extract_roi_from_image(pic_dir, target_dir)
    print("ROIs extracted from images.")

    # 7. Calculate gray values for all ROIs
    dataprocess_utilities.calculate_roi_gray_values(pic_dir, target_dir)
    print("Gray values calculated for all ROIs.")

    # --- Data preprocessing pipeline complete ---
    print("--- Data preprocessing pipeline complete ---")

    # 8. Check CSV shapes
    dataprocess_utilities.load_and_check_csv(target_dir)
    print("CSV shapes checked.")

    # 9. Clean up temporary files
    dataprocess_utilities.cleanup_temp_files(temp_dir)
    print("Temporary files cleaned up.")
    # -----------------------------------