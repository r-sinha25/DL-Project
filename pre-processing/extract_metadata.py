import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Define the input folder and output CSV path
input_folder = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti"  # Replace with your NIfTI folder path
output_csv = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti_metadata_extracted.csv"  # Replace with your desired output path

def extract_metadata(series_folder):
    """Extract metadata from metadata.txt in a series folder."""
    series_instance_id = os.path.basename(series_folder)
    metadata_file = os.path.join(series_folder, "metadata.txt")
    metadata_entries = []

    # Read metadata.txt if it exists
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header line
            for line in lines:
                sop_instance_id, patient_id = line.strip().split("\t")
                metadata_entries.append({
                    "SOPInstanceID": sop_instance_id,
                    "PatientID": patient_id,
                    "SeriesInstanceID": series_instance_id
                })
    return metadata_entries

def process_series_folders(series_folders):
    """Process each series folder and extract metadata."""
    all_metadata = []
    for metadata in tqdm(pool.imap_unordered(extract_metadata, series_folders), total=len(series_folders), desc="Processing Folders"):
        all_metadata.extend(metadata)
    return all_metadata

if __name__ == "__main__":
    # Find all series folders in the input folder
    series_folders = [os.path.join(input_folder, folder) for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    # Use multiprocessing to extract metadata in parallel
    with Pool(cpu_count()) as pool:
        metadata_list = process_series_folders(series_folders)

    # Convert the extracted metadata to a DataFrame
    metadata_df = pd.DataFrame(metadata_list)

    # Save the DataFrame to a CSV file
    metadata_df.to_csv(output_csv, index=False)

    print("Metadata extraction complete! Saved to:", output_csv)
