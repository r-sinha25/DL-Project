import pandas as pd
import numpy as np
from tqdm import tqdm

def verify_hemorrhage_labels(csv_path):
    """
    Comprehensive verification of the hemorrhage labels CSV file
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print("\n=== Basic File Verification ===")
    # Check column names
    expected_columns = {'PatientID', 'SeriesInstanceID', 'hemorrhage_type', 'label'}
    actual_columns = set(df.columns)
    if actual_columns == expected_columns:
        print("✓ All expected columns present")
    else:
        print("! Column mismatch:")
        print(f"  Missing: {expected_columns - actual_columns}")
        print(f"  Extra: {actual_columns - expected_columns}")
    
    # Verify data types
    print("\n=== Data Type Verification ===")
    print(f"PatientID type: {df['PatientID'].dtype}")
    print(f"SeriesInstanceID type: {df['SeriesInstanceID'].dtype}")
    print(f"hemorrhage_type type: {df['hemorrhage_type'].dtype}")
    print(f"label type: {df['label'].dtype}")
    
    # Basic counts
    print("\n=== Basic Counts ===")
    total_rows = len(df)
    total_patients = df['PatientID'].nunique()
    total_series = df['SeriesInstanceID'].nunique()
    print(f"Total rows: {total_rows}")
    print(f"Total unique patients: {total_patients}")
    print(f"Total unique series: {total_series}")
    print(f"Average series per patient: {total_series/total_patients:.2f}")
    
    # Verify expected patient count
    if total_patients == 18938:
        print("✓ Patient count matches expected (18,938)")
    else:
        print(f"! Patient count mismatch. Expected: 18,938, Got: {total_patients}")
    
    # Verify rows per series
    print("\n=== Series Verification ===")
    rows_per_series = df.groupby('SeriesInstanceID').size()
    if rows_per_series.unique().size == 1 and rows_per_series.iloc[0] == 6:
        print("✓ All series have exactly 6 rows")
    else:
        print("! Row count inconsistency detected:")
        print(rows_per_series.value_counts())
    
    # Verify hemorrhage types
    print("\n=== Hemorrhage Type Verification ===")
    expected_types = {'epidural', 'intraparenchymal', 'intraventricular',
                     'subarachnoid', 'subdural', 'any'}
    actual_types = set(df['hemorrhage_type'].unique())
    if actual_types == expected_types:
        print("✓ All hemorrhage types present")
    else:
        print("! Hemorrhage type mismatch:")
        print(f"  Missing: {expected_types - actual_types}")
        print(f"  Extra: {actual_types - expected_types}")
    
    # Verify labels are binary
    if set(df['label'].unique()) <= {0, 1}:
        print("✓ Labels are binary (0 or 1)")
    else:
        print("! Invalid labels found:", set(df['label'].unique()))
    
    # Patient-level statistics
    print("\n=== Patient-Level Statistics ===")
    patient_hemorrhages = df.groupby(['PatientID', 'hemorrhage_type'])['label'].first().unstack()
    
    # Verify hemorrhage distributions
    print("\nHemorrhage Distribution (Patient Level):")
    patients_with_any = patient_hemorrhages['any'].sum()
    print(f"Patients with any hemorrhage: {patients_with_any} ({(patients_with_any/total_patients)*100:.1f}%)")
    
    for h_type in expected_types:
        positive_patients = patient_hemorrhages[h_type].sum()
        percentage = (positive_patients / total_patients) * 100
        percentage_of_positive = (positive_patients / patients_with_any * 100) if h_type != 'any' else None
        
        print(f"\n{h_type}:")
        print(f"  - {positive_patients} patients ({percentage:.1f}% of all patients)")
        if percentage_of_positive is not None:
            print(f"  - {percentage_of_positive:.1f}% of patients with hemorrhage")
    
    # Multiple hemorrhage analysis
    print("\nMultiple Hemorrhage Analysis:")
    hemorrhage_count = patient_hemorrhages[list(expected_types - {'any'})].sum(axis=1)
    for i in range(1, 6):
        count = (hemorrhage_count == i).sum()
        percentage = (count / patients_with_any) * 100
        print(f"Patients with exactly {i} hemorrhage type{'s' if i>1 else ''}: "
              f"{count} ({percentage:.1f}% of positive cases)")
    
    # Consistency checks
    print("\n=== Consistency Checks ===")
    # Check if 'any' matches with other hemorrhage types
    calculated_any = (patient_hemorrhages[list(expected_types - {'any'})].sum(axis=1) > 0)
    matches_any = (calculated_any == patient_hemorrhages['any']).all()
    print("✓ 'any' flag consistency verified") if matches_any else print("! 'any' flag inconsistency detected")
    
    # Sample verification
    print("\n=== Sample Data ===")
    sample_patient = df['PatientID'].iloc[0]
    print(f"Sample data for patient {sample_patient}:")
    print(df[df['PatientID'] == sample_patient])

if __name__ == "__main__":
    csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/hemorrhage_labels_by_series_vertical.csv"
    verify_hemorrhage_labels(csv_path)
