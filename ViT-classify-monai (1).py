#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install monai==1.3.0')


# In[2]:


import monai
print(monai.__version__)


# In[3]:


import os
import torch
import monai
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, SpacingD, OrientationD, ScaleIntensityD,
    ToTensorD, RandFlipD, RandRotateD, RandZoomD, EnsureTypeD
)
from monai.networks.nets import ViT
# from monai.metrics import Accuracy
from torchmetrics.classification import Accuracy
from monai.metrics import compute_confusion_matrix_metric
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from monai.transforms import Resized



# In[4]:


import pandas as pd

def get_label(label_path):
    """
    Load the label CSV file and determine the correct label.
    If no hemorrhage is present, return 'no_hemorrhage'.
    If a hemorrhage is present, return the most specific hemorrhage type.
    """
    try:
        # Read the CSV file
        label_df = pd.read_csv(label_path)
        label_df.columns = label_df.columns.str.strip()  # Clean column names
        label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce").fillna(0).astype(int)  # Ensure labels are integers
        
        # Validate required columns exist
        if "hemorrhage_type" not in label_df.columns or "label" not in label_df.columns:
            print(f"Malformed CSV, missing columns: {label_path}")
            return None

        # Check for specific hemorrhage types
        specific_hemorrhage = label_df[(label_df["label"] == 1) & (label_df["hemorrhage_type"] != "any")]
        if not specific_hemorrhage.empty:
            return specific_hemorrhage["hemorrhage_type"].iloc[0]  # Return first specific hemorrhage type

        # Check for 'any'
        if "any" in label_df["hemorrhage_type"].values:
            any_label = label_df.loc[label_df["hemorrhage_type"] == "any", "label"].iloc[0]
            if any_label == 1:
                return "any"

        # No hemorrhage
        return "no_hemorrhage"

    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return None


# # creating dictionary - only run once

# In[5]:


import os
import pandas as pd
from monai.transforms import LoadImage

# Map hemorrhage types to integers
hemorrhage_types = {
    "intraparenchymal": 0,
    "intraventricular": 1,
    "subarachnoid": 2,
    "subdural": 3,
    "epidural": 4,
    "any": 5  # Add 'any' for completeness
}

# Root data directory
data_root = "/projectnb/ec523kb/projects/hemorrhage-classification/stage_2_train_sorted_nifti_pruned"

# Initialize LoadImage transform to load NIFTI images
load_image = LoadImage(image_only=True)

# Initialize empty list for valid files
train_files = []

# Loop through subfolders starting with 'ID_'
for folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder)
    if os.path.isdir(folder_path) and folder.startswith("ID_"):  # Process only valid folders
        try:
            # Find the NIfTI file and label file in the folder (ignore 'Eq' files)
            image_file = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz") and "Eq" not in f][0]
            label_file = [f for f in os.listdir(folder_path) if f.endswith(".csv")][0]

            # Full paths to image and label
            image_path = os.path.join(folder_path, image_file)
            label_path = os.path.join(folder_path, label_file)

            # Load the image and check its shape
            img = load_image(image_path)
            if img.shape[0] != 6:  # Skip images with 6 channels
                # Parse the labels CSV and convert to numerical format
                df = pd.read_csv(label_path)
                # Get the 'any' column value as the label
                label = int(df.loc[df["hemorrhage_type"] == "any", "label"].values[0])
                train_files.append({"image": image_path, "label": label})
            else:
                print(f"Skipping image with invalid shape: {img.shape}, path: {image_path}")

        except IndexError:
            print(f"Missing NIfTI or CSV file in folder: {folder_path}")
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")

print(f"Total valid files: {len(train_files)}")


# In[6]:


def encode_label(label):
    label_mapping = {
        "intraparenchymal": 0,
        "intraventricular": 1,
        "subarachnoid": 2,
        "subdural": 3,
        "epidural": 4,
        "no_hemorrhage": 5
    }
    # Safely map label or raise an error for debugging
    if label not in label_mapping:
        raise ValueError(f"Unexpected label: {label}")
    return label_mapping[label]


# In[7]:


from sklearn.model_selection import train_test_split

# Split data into train (70%), validation (15%), and test (15%)
train_files, test_val_files = train_test_split(train_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")
print(f"Test samples: {len(test_files)}")


# In[8]:


import torch
import numpy as np
from monai.transforms import MapTransform

# Custom transform to enforce 3 channels
class Ensure3ChannelsD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            if image.shape[0] == 6:  # If 6 channels, take the first 3 channels
                image = image[:3, ...]
            elif image.shape[0] == 1:  # If single channel, repeat it 3 times
                image = torch.cat([image] * 3, dim=0)
            elif image.shape[0] > 3:  # In any other case with >3 channels, truncate to 3
                image = image[:3, ...]
            elif image.shape[0] < 3:  # If less than 3 channels, repeat to reach 3
                image = torch.cat([image] * 3, dim=0)[:3, ...]
            data[key] = image
        return data

class SkipInvalidScansD(MapTransform):
    """
    Custom transform to skip scans with invalid channels (e.g., 6 channels).
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            if image.shape[0] == 6:  # Skip this image
                raise ValueError(f"Invalid image with shape {image.shape}, skipping.")
        return data


# In[9]:


from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, Resized, ScaleIntensityD, ToTensorD, LambdaD, RepeatChannelD, CastToTypeD
)

# Training Transforms
train_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    RepeatChannelD(keys=["image"], repeats=3),
    ScaleIntensityD(keys=["image"]),
    ToTensorD(keys=["image", "label"])             # Convert both image and label to Tensor
])

# Validation Transforms
val_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    RepeatChannelD(keys=["image"], repeats=3),
    ScaleIntensityD(keys=["image"]),
    ToTensorD(keys=["image", "label"])
])

# Test Transforms
test_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    RepeatChannelD(keys=["image"], repeats=3),
    ScaleIntensityD(keys=["image"]),
    ToTensorD(keys=["image", "label"])
])


# In[10]:


for sample in train_files:
    print(f"Label: {sample['label']},  Type: {type(sample['label'])}")
    break


# In[ ]:


# Create datasets
train_ds = CacheDataset(train_files, transform=train_transforms, cache_rate=0.8)
val_ds = CacheDataset(val_files, transform=train_transforms, cache_rate=0.8)
test_ds = CacheDataset(test_files, transform=train_transforms, cache_rate=0.8)


# In[ ]:


for sample in train_ds:
    if int(sample["image"].shape[0]) == 6:
        print("AHHHH")
#     print(sample["image"].shape)  # Should print [3, 128, 128, 128]
    


# In[ ]:


from monai.data import pad_list_data_collate

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=pad_list_data_collate)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=pad_list_data_collate)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=pad_list_data_collate)


# In[ ]:


from monai.networks.nets import ViT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated ViT configuration for MONAI 1.4.0
model = ViT(
    in_channels=3,                   # Input image has 3 channels
    img_size=(128, 128, 128),        # Input image size
    patch_size=(16, 16, 16),         # Patch size for splitting the image
    num_layers=24,                   # 24 transformer blocks
    classification=True,             # Enable classification
    num_classes=5,                   # Number of output classes
    spatial_dims=3,                  # 3D input data
    hidden_size=768,                 # Transformer hidden size
    mlp_dim=3072,                    # MLP dimension
    num_heads=12                     # Number of attention heads
).to(device)

# model = ViT(
#         in_channels=3,
#         img_size=(128,128,128),
#         proj_type='conv',
#         pos_embed_type='sincos',
#         classification=True
# ).to(device)


# In[ ]:


loss_function = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
accuracy_metric = Accuracy(task="multiclass", num_classes=5)  # Adjust num_classes


# In[ ]:


from tqdm import tqdm

model.train()
max_epochs = 10
best_val_loss = float("inf")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        
        # Ensure labels are a Tensor
        if isinstance(labels, tuple):
            labels = labels[0]
        
        optimizer.zero_grad()
        outputs = model(inputs)

        # Fix model output
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Compute loss
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch["image"].to(device), val_batch["label"].to(device)

            # Ensure labels are Tensor
            if isinstance(val_labels, tuple):
                val_labels = val_labels[0]

            val_outputs = model(val_inputs)

            # Fix model output
            if isinstance(val_outputs, tuple):
                val_outputs = val_outputs[0]

            val_loss += loss_function(val_outputs, val_labels).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")


# In[ ]:


torch.save(model.state_dict(), "vision_transformer_ct_classifier.pth")


# In[ ]:




