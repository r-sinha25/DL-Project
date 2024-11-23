import pandas as pd
import os
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import numpy as np

def process_single_series(series_id, df, main_folder_path):
    """
    Process a single series and create its hemorrhage labels CSV
    """
    try:
        # Get data for this series
        series_data = df[df['SeriesInstanceID'] == series_id]
        
        if len(series_data) == 0:
            return False
            
        # Create the label data
        label_data = {
            'hemorrhage_type': [
                'epidural',
                'intraparenchymal',
                'intraventricular',
                'subarachnoid',
                'subdural',
                'any'
            ],
            'label': [
                series_data[series_data['Hemorrhage_type'] == 'epidural']['Label'].iloc[0],
                series_data[series_data['Hemorrhage_type'] == 'intraparenchymal']['Label'].iloc[0],
                series_data[series_data['Hemorrhage_type'] == 'intraventricular']['Label'].iloc[0],
                series_data[series_data['Hemorrhage_type'] == 'subarachnoid']['Label'].iloc[0],
                series_data[series_data['Hemorrhage_type'] == 'subdural']['Label'].iloc[0],
                series_data[series_data['Hemorrhage_type'] == 'any']['Label'].iloc[0]
            ]
        }
        
        # Create DataFrame with hemorrhage labels
        labels_df = pd.DataFrame(label_data)
        
        # Define the folder path and output path
        series_folder = os.path.join(main_folder_path, series_id)
        output_path = os.path.join(series_folder, 'hemorrhage_labels.csv')
        
        # Check if folder exists and save
        if os.path.exists(series_folder):
            labels_df.to_csv(output_path, index=False)
            return True
            
    except Exception as e:
        print(f"\nError processing series {series_id}: {str(e)}")
        return False
    
    return False

def create_series_labels_parallel(merged_csv_path, main_folder_path):
    """
    Create hemorrhage label CSV files for each SeriesInstanceID folder using parallel processing
    """
    # Read the merged CSV file
    print("Reading CSV file...")
    df = pd.read_csv(merged_csv_path)
    
    # Get unique SeriesInstanceIDs
    series_ids = df['SeriesInstanceID'].unique()
    total_series = len(series_ids)
    print(f"Found {total_series} unique SeriesInstanceID folders to process")
    
    # Calculate optimal chunk size based on number of CPU cores
    num_cores = mp.cpu_count()
    chunk_size = max(1, min(100, total_series // (num_cores * 4)))
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_series, df=df, main_folder_path=main_folder_path)
    
    # Initialize multiprocessing with 'fork' start method (recommended for M1 Macs)
    mp.set_start_method('fork', force=True)
    
    print(f"\nStarting parallel processing using {num_cores} cores...")
    
    try:
        # Create a pool of workers
        with mp.Pool(processes=num_cores) as pool:
            # Process series_ids with progress bar
            results = list(tqdm(
                pool.imap(process_func, series_ids, chunksize=chunk_size),
                total=total_series,
                desc="Processing series",
                unit="series"
            ))
        
        # Calculate statistics
        processed = sum(results)
        skipped = total_series - processed
        
        print("\n=== Processing Complete ===")
        print(f"Total series processed successfully: {processed}")
        print(f"Total series skipped or failed: {skipped}")
        
        # Verify a random series
        if processed > 0:
            random_series = np.random.choice(series_ids)
            random_labels_path = os.path.join(main_folder_path, random_series, 'hemorrhage_labels.csv')
            if os.path.exists(random_labels_path):
                print(f"\nSample labels for random series {random_series}:")
                print(pd.read_csv(random_labels_path))
                
    except Exception as e:
        print(f"\nError during parallel processing: {str(e)}")
        
def verify_processing(main_folder_path):
    """
    Verify the processing results
    """
    print("\nVerifying results...")
    all_series = [f.name for f in os.scandir(main_folder_path) if f.is_dir()]
    processed_count = 0
    
    for series_id in tqdm(all_series, desc="Verifying", unit="series"):
        label_path = os.path.join(main_folder_path, series_id, 'hemorrhage_labels.csv')
        if os.path.exists(label_path):
            processed_count += 1
            
    print(f"\nVerification complete: {processed_count}/{len(all_series)} series have label files")

if __name__ == "__main__":
    merged_csv_path = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_final2_sorted_copy.csv"
    main_folder_path = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_sorted_nifti_copy"
    
    # Process the files
    create_series_labels_parallel(merged_csv_path, main_folder_path)
    
    # Verify the results
    verify_processing(main_folder_path)
