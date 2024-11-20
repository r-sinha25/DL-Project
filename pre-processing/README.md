# Pre-Processing Training Data

The original data from the Kaggle competition contain single slice image of the CT scan in the DICOM (.dcm) format. The pre-processing pipeline includes converting DICOM files to NIfTI format, extracting metadata, and merging labels for use in deep learning models.

## Overview

The pipeline prepares the training data by:
1. Grouping DICOM files by SeriesInstanceUID (NIFTI works by combining all DICOM files into a single 3-D imaging. DICOM files that share the same SeriesInstanceUID belongs to the same image)
2. Converting grouped DICOM files into NIfTI format.
3. Extracting metadata for each SeriesInstanceUID (this allows us to perserve critical metadata info such as PatientID and SOPInstanceID)
4. Merging extracted metadata with the provided labels (stage_2_train.csv).
5. Generate final CSV file for analysis

## Training Data Result

Based on our analysis of the training data, we determined that: 