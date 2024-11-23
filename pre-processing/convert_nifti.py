import os
import subprocess
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import shutil

# Define the input and output paths
input_folder = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_small"  # Replace with the existing grouped folder path
output_folder = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti_test"  # Replace with the new output folder for NIfTI files

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def convert_series(args):
    """Convert a single series to NIfTI."""
    series_uid, series_path = args

    # Define the new series folder in the converted directory
    new_series_folder = os.path.join(output_folder, series_uid)
    os.makedirs(new_series_folder, exist_ok=True)

    # Copy the metadata file
    metadata_src = os.path.join(series_path, "metadata.txt")
    metadata_dest = os.path.join(new_series_folder, "metadata.txt")
    if os.path.exists(metadata_src):
        shutil.copy(metadata_src, metadata_dest)

    try:
        # Verify .dcm files in the input folder
        dcm_files = [f for f in os.listdir(series_path) if f.endswith(".dcm")]
        if not dcm_files:
            print(f"No .dcm files found in {series_path}. Skipping {series_uid}.")
            return False

        # Copy .dcm files to the new output folder for processing
        for dcm_file in dcm_files:
            shutil.copy(os.path.join(series_path, dcm_file), os.path.join(new_series_folder, dcm_file))

        # Run dcm2niix to convert DICOM to NIfTI
        result = subprocess.run(
            [
                "dcm2niix", "-z", "y", "-f", series_uid, "-o", new_series_folder, new_series_folder, "-v", "n"
            ],
            capture_output=True,
            text=True,
        )

        # Check for errors during conversion
        if result.returncode != 0:
            print(f"Error converting {series_uid} to NIfTI:\n{result.stderr}")
            return False
        else:
            print(f"Converted {series_uid} to NIfTI successfully:\n{result.stdout}")

            # Delete .dcm files from the new output folder after successful conversion
            for dcm_file in dcm_files:
                os.remove(os.path.join(new_series_folder, dcm_file))
            print(f"Deleted .dcm files for {series_uid} in the output folder.")
            return True
    except Exception as e:
        print(f"Error processing series {series_uid}: {e}")
        return False

def get_series_folders(input_folder):
    """Retrieve all series folders from the input directory."""
    return [
        (folder, os.path.join(input_folder, folder))
        for folder in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, folder))
    ]

def process_all_series(input_folder):
    """Process all series in the input directory using parallel processing."""
    series_folders = get_series_folders(input_folder)
    print(f"Found {len(series_folders)} series folders to process.")

    # Use multiprocessing to process series in parallel
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(convert_series, series_folders), total=len(series_folders), desc="Converting Series"))

if __name__ == "__main__":
    # Set spawn method for macOS compatibility
    set_start_method("spawn", force=True)

    # Start the conversion process
    process_all_series(input_folder)
    print("All series converted to NIfTI.")
