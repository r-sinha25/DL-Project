import pandas as pd

# Load the original CSV file
# Replace 'path_to_stage_2_train.csv' with the actual path to your file
input_file = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"
df = pd.read_csv(input_file)

# Split the 'ID' column into 'SOPInstanceID' and 'Hemorrhage_type'
df[['SOPInstanceID', 'Hemorrhage_type']] = df['ID'].str.rsplit('_', n=1, expand=True)

# Rearrange the columns to match the desired format
transformed_df = df[['SOPInstanceID', 'Hemorrhage_type', 'Label']]

# Save the transformed DataFrame to a new CSV
output_file = "/Volumes/SanDisk Extreme 55AE Media/rsna-intracranial-hemorrhage-detection/stage_2_train_transformed1.csv"  # Replace with your desired output path
transformed_df.to_csv(output_file, index=False)

print("Transformation complete! New file saved as:", output_file)
