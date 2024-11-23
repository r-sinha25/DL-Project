import pandas as pd
import numpy as np
from tqdm import tqdm

def prune_dataset(input_csv_path, output_csv_path, target_percentage=0.20):
    """
    Prune the dataset while maintaining distributions and keeping only single hemorrhage cases
    """
    print(f"Reading CSV file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # First, get patient-level hemorrhage information
    patient_hemorrhages = df.groupby(['PatientID', 'hemorrhage_type'])['label'].first().unstack()
    
    # Calculate hemorrhage counts per patient (excluding 'any')
    hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    hemorrhage_count = patient_hemorrhages[hemorrhage_types].sum(axis=1)
    
    # Separate patients into categories
    negative_patients = patient_hemorrhages[patient_hemorrhages['any'] == 0].index
    single_hemorrhage_patients = patient_hemorrhages[
        (patient_hemorrhages['any'] == 1) & (hemorrhage_count == 1)
    ].index
    
    # Get hemorrhage distribution for single hemorrhage patients
    single_hemorrhage_distribution = {}
    for h_type in hemorrhage_types:
        single_type_patients = patient_hemorrhages[
            (hemorrhage_count == 1) &
            (patient_hemorrhages[h_type] == 1)
        ].index
        single_hemorrhage_distribution[h_type] = list(single_type_patients)
        
    print("\n=== Original Distribution ===")
    print(f"Negative patients: {len(negative_patients)}")
    print("\nSingle hemorrhage patients by type:")
    for h_type, patients in single_hemorrhage_distribution.items():
        print(f"{h_type}: {len(patients)}")
    
    # Calculate target numbers
    total_target = int(len(patient_hemorrhages) * target_percentage)
    target_negative = int(total_target * 0.596)  # Maintain 59.6% negative
    target_positive = total_target - target_negative
    
    # Select negative patients
    selected_negative = np.random.choice(negative_patients, target_negative, replace=False)
    
    # Select positive patients maintaining original ratios
    total_single = sum(len(patients) for patients in single_hemorrhage_distribution.values())
    selected_positive = []
    
    for h_type, patients in single_hemorrhage_distribution.items():
        # Calculate target for this type based on original ratio
        type_ratio = len(patients) / total_single
        type_target = int(target_positive * type_ratio)
        
        # Select patients for this type
        if len(patients) > 0:
            selected = np.random.choice(
                patients,
                min(type_target, len(patients)),
                replace=False
            )
            selected_positive.extend(selected)
    
    # Combine all selected patients
    selected_patients = list(selected_negative) + selected_positive
    
    # Create new dataframe with only selected patients
    print("\nCreating pruned dataset...")
    pruned_df = df[df['PatientID'].isin(selected_patients)]
    
    # Save pruned dataset
    pruned_df.to_csv(output_csv_path, index=False)
    
    # Print final statistics
    print("\n=== Pruned Dataset Statistics ===")
    final_patients = pruned_df['PatientID'].nunique()
    final_series = pruned_df['SeriesInstanceID'].nunique()
    print(f"Total patients: {final_patients}")
    print(f"Total series: {final_series}")
    
    # Calculate final distributions
    final_patient_hemorrhages = pruned_df.groupby(['PatientID', 'hemorrhage_type'])['label'].first().unstack()
    final_negative = (final_patient_hemorrhages['any'] == 0).sum()
    print(f"\nNegative patients: {final_negative} ({final_negative/final_patients*100:.1f}%)")
    
    print("\nPositive patients by type (single hemorrhage only):")
    for h_type in hemorrhage_types:
        count = ((hemorrhage_count == 1) & (final_patient_hemorrhages[h_type] == 1)).sum()
        print(f"{h_type}: {count} ({count/final_patients*100:.1f}%)")
    
    return pruned_df, selected_patients

if __name__ == "__main__":
    input_csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/hemorrhage_labels_by_series_vertical.csv"
    output_csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/hemorrhage_labels_pruned.csv"
    
    pruned_df, selected_patients = prune_dataset(input_csv_path, output_csv_path)
