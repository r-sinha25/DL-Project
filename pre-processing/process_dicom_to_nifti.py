import os
import pydicom
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import warnings

# Suppress warnings from pydicom
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

# Define input and output paths
input_folder = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train"
output_folder = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process a single series
def process_series(args):
    series_uid, file_paths = args

    # Create a folder for each SeriesInstanceUID
    series_folder = os.path.join(output_folder, series_uid)
    os.makedirs(series_folder, exist_ok=True)

    # Create metadata.txt file
    metadata_file_path = os.path.join(series_folder, "metadata.txt")
    with open(metadata_file_path, "w") as metadata_file:
        metadata_file.write("SOPInstanceUID\tPatientID\n")
        for file_path in file_paths:
            try:
                # Read DICOM metadata
                ds = pydicom.dcmread(file_path)
                sop_instance_uid = ds.SOPInstanceUID
                patient_id = ds.PatientID

                # Write metadata to the file
                metadata_file.write(f"{sop_instance_uid}\t{patient_id}\n")

                # Copy the .dcm file to the series folder
                dest_path = os.path.join(series_folder, os.path.basename(file_path))
                shutil.copy(file_path, dest_path)
                print(f"Copied {file_path} to {dest_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Print the grouping status
    print(f"{len(file_paths)} .dcm files grouped into folder: {series_uid}")

# Function to group files by SeriesInstanceUID
def group_files():
    series_dict = defaultdict(list)

    print(f"Scanning input folder: {input_folder}")
    if not os.path.exists(input_folder):
        print("Error: Input folder does not exist!")
        return series_dict

    print("Grouping files by SeriesInstanceUID...")
    total_files = 0

    for root, _, files in tqdm(os.walk(input_folder), desc="Scanning DICOM files"):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                try:
                    # Read DICOM metadata
                    ds = pydicom.dcmread(file_path)
                    series_uid = ds.SeriesInstanceUID
                    series_dict[series_uid].append(file_path)
                    total_files += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    print(f"Total files scanned: {total_files}")
    print(f"Total series identified: {len(series_dict)}")
    return series_dict

if __name__ == "__main__":
    # Set spawn method for macOS compatibility
    set_start_method("spawn", force=True)

    # Group files
    series_dict = group_files()

    # Process series in parallel
    print("Processing each series in parallel...")
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_series, series_dict.items()), total=len(series_dict), desc="Processing Series"))

    print("All files grouped into series folders.")
