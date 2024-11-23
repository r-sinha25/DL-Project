import pandas as pd

def merge_medical_data(hemorrhage_file, metadata_file, output_file):
    """
    Merge hemorrhage data with patient metadata based on SOPInstanceID.
    
    Parameters:
    hemorrhage_file (str): Path to the hemorrhage CSV file
    metadata_file (str): Path to the metadata CSV file
    output_file (str): Path for the output merged CSV file
    """
    # Read the CSV files
    hemorrhage_df = pd.read_csv(hemorrhage_file)
    metadata_df = pd.read_csv(metadata_file)
    
    # Ensure column names are stripped of any whitespace
    hemorrhage_df.columns = hemorrhage_df.columns.str.strip()
    metadata_df.columns = metadata_df.columns.str.strip()
    
    # Merge the dataframes on SOPInstanceID
    merged_df = pd.merge(
        hemorrhage_df,
        metadata_df[['SOPInstanceID', 'PatientID', 'SeriesInstanceID']],
        on='SOPInstanceID',
        how='left'
    )
    
    # Verify the merge results
    initial_rows = len(hemorrhage_df)
    final_rows = len(merged_df)
    
    # Print verification information
    print(f"Original hemorrhage entries: {initial_rows}")
    print(f"Final merged entries: {final_rows}")
    
    # Check for any missing matches
    missing_matches = merged_df[merged_df['PatientID'].isna()]['SOPInstanceID'].unique()
    if len(missing_matches) > 0:
        print(f"\nWarning: {len(missing_matches)} SOPInstanceIDs didn't find matches in metadata file")
        print("First few unmatched IDs:", missing_matches[:5])
    
    # Verify the structure
    patient_counts = merged_df.groupby('PatientID').size()
    print("\nSample of entries per patient:")
    print(patient_counts.head())
    
    # Save the merged dataset
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged data saved to {output_file}")
    
    return merged_df

# Example usage
if __name__ == "__main__":
    hemorrhage_file = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_transformed1.csv"
    metadata_file = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti_metadata_extracted.csv"
    output_file = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_final2.csv"
    
    merged_data = merge_medical_data(hemorrhage_file, metadata_file, output_file)
