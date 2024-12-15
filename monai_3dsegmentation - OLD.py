import os
import monai
import numpy as np
import torch
import pandas as pd
from monai import transforms
from monai.data import Dataset, DataLoader
from monai.optimizers import Novograd
from monai.utils import set_determinism
import multiprocessing


def load_patient_files(patient_folder):
    """
    Extract image and label files for a patient.

    Args:
        patient_folder (str): Path to the patient's folder.

    Returns:
        dict: Dictionary with paths for image and labels.
    """
    patient_id = os.path.basename(patient_folder)

    print(f"Files in folder {patient_folder}: {os.listdir(patient_folder)}")

    nifti_files = [f for f in os.listdir(patient_folder) if f.endswith(".nii.gz")]
    if not nifti_files:
        print(f"Warning: No NIfTI file found in {patient_folder}")
        return None
    nifti_file = os.path.join(patient_folder, nifti_files[0])

    label_file = os.path.join(patient_folder, "hemorrhage_labels.csv")
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found: {label_file}")
        return None

    try:
        labels = pd.read_csv(label_file)["label"].values
        labels = labels.astype(np.float32)
        print(f"Labels loaded for {label_file}: {labels}")
    except Exception as e:
        print(f"Error reading label file {label_file}: {e}")
        return None

    return {"image": nifti_file, "label": labels}


def create_data_list(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The dataset directory does not exist: {data_dir}")
    print(f"Contents of the dataset directory: {os.listdir(data_dir)}")

    patient_folders = [
        os.path.join(data_dir, folder) for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ]

    print(f"Detected patient folders: {patient_folders}")
    if not patient_folders:
        raise ValueError("No patient folders detected in the dataset directory.")

    data_list = []
    for folder in patient_folders:
        print(f"Processing folder: {folder}")
        try:
            data = load_patient_files(folder)
            if data:
                data_list.append(data)
        except Exception as e:
            print(f"Skipping patient folder {folder}: {e}")

    return data_list


class UNetWithClassificationHead(torch.nn.Module):
    def __init__(self, num_classes=6):  # Updated num_classes to match the dataset
        super().__init__()
        self.unet = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=16,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        features = self.unet(x)
        pooled = torch.nn.AdaptiveAvgPool3d(1)(features)
        pooled = torch.flatten(pooled, 1)
        print(f"Pooled features shape: {pooled.shape}")  # Debugging
        return self.fc(pooled)


def main():
    set_determinism(seed=42)

    train_data_dir = "/Users/rupalisinha/Desktop/Deep Learning/stage_2_train_sorted_nifti_pruned"

    train_files = create_data_list(train_data_dir)
    print(f"Number of training samples: {len(train_files)}")

    if not train_files:
        raise ValueError("No valid training samples found. Please check the dataset structure.")

    train_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: np.expand_dims(x, axis=0)),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        transforms.OrientationD(keys=["image"], axcodes="RAS"),
        transforms.ScaleIntensityd(keys=["image"]),
        transforms.Resized(keys=["image"], spatial_size=(128, 128, 128)),
        transforms.ToTensord(keys=["image"])
    ])

    class CustomDataset(Dataset):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            data["label"] = torch.tensor(train_files[index]["label"], dtype=torch.float32)
            return data

    train_ds = CustomDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNetWithClassificationHead(num_classes=6).to(device)  # Updated num_classes

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = Novograd(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_data in train_loader:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)  # No need to convert to long

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "ich_classification_model.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
