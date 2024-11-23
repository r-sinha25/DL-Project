import pandas as pd
import numpy as np
from tqdm import tqdm

def restructure_and_analyze_labels(input_csv_path, output_csv_path):
    """
    Analyze at patient level first, then create series-level labels
    """
    print("Reading original CSV file...")
    df = pd.read_csv(input_csv_path)
    
    # First analyze at patient level
    print("\n=== Original Data Analysis (Patient Level) ===")
    patient_hemorrhages = df.groupby(['PatientID', 'Hemorrhage_type'])['Label'].max().reset_index()
    
    # Create pivot table for patient-level verification
    patient_pivot = patient_hemorrhages.pivot(
        index='PatientID',
        columns='Hemorrhage_type',
        values='Label'
    ).fillna(0)
    
    total_patients = len(patient_pivot)
    
    print(f"\nPatient-Level Statistics:")
    print(f"Total unique patients: {total_patients}")
    
    # Verify hemorrhage distributions
    hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular',
                       'subarachnoid', 'subdural', 'any']
    
    print("\nHemorrhage Distribution (Patient Level):")
    for h_type in hemorrhage_types:
        positive_patients = patient_pivot[h_type].sum()
        percentage = (positive_patients / total_patients) * 100
        percentage_of_positive = (positive_patients / patient_pivot['any'].sum()) * 100
        print(f"{h_type}:")
        print(f"  - {positive_patients} patients ({percentage:.1f}% of all patients)")
        if h_type != 'any':
            print(f"  - {percentage_of_positive:.1f}% of patients with hemorrhage")
    
    # Multiple hemorrhage analysis
    hemorrhage_count = patient_pivot[hemorrhage_types[:-1]].sum(axis=1)  # Exclude 'any'
    positive_patients = patient_pivot['any'].sum()
    
    print("\nMultiple Hemorrhage Analysis:")
    for i in range(1, 6):
        count = (hemorrhage_count == i).sum()
        percentage = (count / positive_patients) * 100
        print(f"Patients with exactly {i} hemorrhage type{'s' if i>1 else ''}: "
              f"{count} ({percentage:.1f}% of positive cases)")
    
    # Now create the series-level labels file
    print("\nCreating series-level labels file...")
    series_data = []
    
    # Group by series first to get unique series
    series_groups = df.groupby('SeriesInstanceID')
    
    for series_id, series_group in tqdm(series_groups, desc="Processing series"):
        patient_id = series_group['PatientID'].iloc[0]  # Get patient ID for this series
        
        # Get hemorrhage labels for this patient
        patient_labels = patient_pivot.loc[patient_id]
        
        # Create 6 rows for this series
        for h_type in hemorrhage_types:
            series_data.append({
                'PatientID': patient_id,
                'SeriesInstanceID': series_id,
                'hemorrhage_type': h_type,
                'label': int(patient_labels[h_type])
            })
    
    # Create final dataframe
    restructured_df = pd.DataFrame(series_data)
    
    # Save restructured data
    restructured_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved restructured data to: {output_csv_path}")
    
    # Verify the output file
    print("\nVerifying output file...")
    output_df = pd.read_csv(output_csv_path)
    total_series = output_df['SeriesInstanceID'].nunique()
    
    print(f"\nOutput File Statistics:")
    print(f"Total series: {total_series}")
    print(f"Average series per patient: {total_series/total_patients:.2f}")
    
    rows_per_series = output_df.groupby('SeriesInstanceID').size()
    if rows_per_series.unique().size == 1 and rows_per_series.iloc[0] == 6:
        print("✓ Verified: Each series has exactly 6 rows (one per hemorrhage type)")
    else:
        print("! Warning: Not all series have exactly 6 rows")
        
    return restructured_df, patient_pivot

if __name__ == "__main__":
    input_csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/stage_2_train_final2_sorted.csv"
    output_csv_path = "/Volumes/SanDisk Extreme 55AE Media/final/hemorrhage_labels_by_series_vertical.csv"
    
    restructured_df = restructure_and_analyze_labels(input_csv_path, output_csv_path)
