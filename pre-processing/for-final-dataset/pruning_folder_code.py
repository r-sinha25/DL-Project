import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from functools import partial
import random
from concurrent.futures import ProcessPoolExecutor
import time

def verify_folder_contents(src_path, dst_path):
    """Verify that all original contents are copied correctly"""
    try:
        src_files = set(os.listdir(src_path))
        dst_files = set(os.listdir(dst_path))
        
        # Check if all source files exist in destination
        return src_files.issubset(dst_files)
    except Exception as e:
        print(f"Error verifying folders: {str(e)}")
        return False

def create_label_file(series_id, labels_df, folder_path):
    """Create hemorrhage labels CSV file for a series"""
    series_labels = labels_df[labels_df['SeriesInstanceID'] == series_id]
    if not series_labels.empty:
        label_file_path = os.path.join(folder_path, 'hemorrhage_labels.csv')
        series_labels[['hemorrhage_type', 'label']].to_csv(label_file_path, index=False)
        return True
    return False

def copy_single_series(args):
    """Copy a single series folder and add labels"""
    series_id, source_dir, target_dir, labels_df = args
    try:
        source_path = os.path.join(source_dir, series_id)
        target_path = os.path.join(target_dir, series_id)
        
        # Copy folder if it exists
        if os.path.exists(source_path):
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
            
            # Verify copy
            if verify_folder_contents(source_path, target_path):
                # Add labels file
                if create_label_file(series_id, labels_df, target_path):
                    return True, series_id
        
        return False, series_id
    except Exception as e:
        print(f"\nError processing {series_id}: {str(e)}")
        return False, series_id

def process_dataset(input_csv_path, source_dir, target_dir):
    """Process the dataset with deduplication and parallel copying"""
    print("Reading CSV file and preparing data...")
    df = pd.read_csv(input_csv_path)
    
    # Deduplicate patients by keeping one random series per patient
    print("\nDeduplicating patients...")
    patient_series = df.groupby('PatientID')['SeriesInstanceID'].agg(list).to_dict()
    selected_series = [random.choice(series_list) for series_list in patient_series.values()]
    
    # Filter dataframe to keep only selected series
    df_deduplicated = df[df['SeriesInstanceID'].isin(selected_series)]
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = [(series_id, source_dir, target_dir, df_deduplicated)
                 for series_id in selected_series]
    
    # Process in parallel
    print("\nCopying files in parallel...")
    num_cores = mp.cpu_count()
    successful = []
    failed = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(
            executor.map(copy_single_series, args_list),
            total=len(args_list),
            desc="Copying folders"
        ))
    
    # Process results
    for success, series_id in results:
        if success:
            successful.append(series_id)
        else:
            failed.append(series_id)
    
    # Generate statistics
    print("\n=== Final Dataset Statistics ===")
    
    # Basic counts
    print("\nBasic Counts:")
    print(f"Successfully copied series: {len(successful)}")
    print(f"Failed copies: {len(failed)}")
    
    # Get final dataframe of copied data
    final_df = df_deduplicated[df_deduplicated['SeriesInstanceID'].isin(successful)]
    
    # Patient and hemorrhage statistics
    print("\nPatient Statistics:")
    total_patients = final_df['PatientID'].nunique()
    print(f"Total unique patients: {total_patients}")
    
    # Hemorrhage distribution
    print("\nHemorrhage Distribution:")
    patient_hemorrhages = final_df.groupby(['PatientID', 'hemorrhage_type'])['label'].first().unstack()
    
    positive_patients = (patient_hemorrhages['any'] == 1).sum()
    negative_patients = (patient_hemorrhages['any'] == 0).sum()
    print(f"Positive patients: {positive_patients} ({positive_patients/total_patients*100:.1f}%)")
    print(f"Negative patients: {negative_patients} ({negative_patients/total_patients*100:.1f}%)")
    
    hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular',
                       'subarachnoid', 'subdural']
    
    print("\nHemorrhage Type Breakdown:")
    for h_type in hemorrhage_types:
        count = (patient_hemorrhages[h_type] == 1).sum()
        percentage = count/total_patients*100
        percentage_of_positive = count/positive_patients*100
        print(f"{h_type}:")
        print(f"  - {count} patients ({percentage:.1f}% of all patients)")
        print(f"  - {percentage_of_positive:.1f}% of positive patients")
    
    # File size statistics
    print("\nStorage Statistics:")
    total_size = 0
    file_counts = {'nifti': 0, 'metadata': 0, 'labels': 0}
    
    for series_id in tqdm(successful, desc="Calculating sizes"):
        folder_path = os.path.join(target_dir, series_id)
        if os.path.exists(folder_path):
            # Calculate size
            folder_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(folder_path)
                for filename in filenames
            )
            total_size += folder_size
            
            # Count file types
            for file in os.listdir(folder_path):
                if file.endswith('.nii.gz'):
                    file_counts['nifti'] += 1
                elif 'metadata' in file.lower():
                    file_counts['metadata'] += 1
                elif file == 'hemorrhage_labels.csv':
                    file_counts['labels'] += 1
    
    print(f"Total dataset size: {total_size / (1024*1024*1024):.2f} GB")
    print("\nFile counts:")
    for file_type, count in file_counts.items():
        print(f"{file_type}: {count} files")
    
    return final_df, successful, failed

if __name__ == "__main__":
    input_csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/hemorrhage_labels_pruned.csv"
    source_dir = "/Volumes/SanDisk Extreme 55AE Media/final/stage_2_train_sorted_nifti"
    target_dir = "/Volumes/SanDisk Extreme 55AE Media/final/stage_2_train_sorted_nifti_pruned"
    
    final_df, successful, failed = process_dataset(input_csv_path, source_dir, target_dir)
