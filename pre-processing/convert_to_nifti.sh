#!/bin/bash

# Define input and output folders
INPUT_FOLDER="/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted"   # Replace with your grouped folder path
OUTPUT_FOLDER="/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti"       # Replace with the new output folder

# Ensure the output folder exists
mkdir -p "$OUTPUT_FOLDER"

# Function to process a single folder
process_folder() {
    SERIES_FOLDER="$1"
    SERIES_UID=$(basename "$SERIES_FOLDER")
    echo "Processing $SERIES_UID..."

    # Create a corresponding folder in the output directory
    OUTPUT_SERIES_FOLDER="$OUTPUT_FOLDER/$SERIES_UID"
    mkdir -p "$OUTPUT_SERIES_FOLDER"

    # Copy metadata.txt to the new folder
    if [ -f "$SERIES_FOLDER/metadata.txt" ]; then
        cp "$SERIES_FOLDER/metadata.txt" "$OUTPUT_SERIES_FOLDER/"
    fi

    # Run dcm2niix on the series folder, suppress warnings
    dcm2niix -z y -f "$SERIES_UID" -o "$OUTPUT_SERIES_FOLDER" "$SERIES_FOLDER" >/dev/null 2>&1

    # Check if NIfTI file was created
    if [ $? -eq 0 ]; then
        echo "Successfully converted $SERIES_UID to NIfTI."
        # Optionally delete .dcm files in the output folder after successful conversion
        find "$OUTPUT_SERIES_FOLDER" -type f -name "*.dcm" -delete
    else
        echo "Failed to convert $SERIES_UID to NIfTI."
    fi
}

export -f process_folder  # Export the function for parallel to use
export OUTPUT_FOLDER      # Export the variable for parallel to access

# Use parallel with a progress bar to process each folder
find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d | parallel --bar -j "$(sysctl -n hw.ncpu)" process_folder
