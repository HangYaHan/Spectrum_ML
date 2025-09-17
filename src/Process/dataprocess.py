import dataprocess_utilities
import os
import config


def data_preprocessing_pipeline(pic_dir, spec_dir, target_dir):
    """
    Main pipeline for data preprocessing and feature extraction.
    Each step should be implemented as a separate function.
    """
    # 1. Rename and match files
    # rename_and_match_files(pic_dir, spec_dir)

    # 2. Remove header lines from spectrum files
    # remove_header_lines(spec_path, spec_clean_path, num_lines=14)

    # 3. Convert spectrum data to integer
    # convert_spectrum_to_int(spec_clean_path, spec_int_path)

    # 4. Filter spectrum to visible wavelength range
    # filter_wavelength_range(spec_int_path, spec_visible_path, min_wavelength=400, max_wavelength=800)

    # 5. Extract ROIs from images
    # extract_roi_from_image(image_path, roi_config_path, num_regions)

    # 6. Calculate gray values for all ROIs
    # calculate_roi_gray_values(roi_config_path, pic_dir, grey_csv_path)

    # 7. Convert spectrum to CSV
    # spectrum_to_csv(spec_visible_path, spectrum_csv_path)

    # 8. Check CSV shapes
    # load_and_check_csv(grey_csv_path)
    # load_and_check_csv(spectrum_csv_path)

    # 9. Calculate CCT for all spectra
    # calculate_cct_for_spectra(spectrum_csv_path, cct_csv_path, step=1, wavelength_s=400, wavelength_e=800)

    # 10. Count CCT above threshold
    # count_cct_above_threshold(cct_csv_path, threshold=8000)

    pass  # Remove after implementing each step

if __name__ == "__main__":
    data_preprocessing_pipeline(config.original_pics_folder, config.original_specs_folder, ".\\ProcessedData\\FinalCSV")