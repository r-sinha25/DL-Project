#!/usr/bin/env python
# coding: utf-8

# In[60]:


import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
import torch.nn.parallel as parallel
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from IPython.display import display


# In[61]:


# Vision Transformer Components
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, embed_dim))
        
    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


# In[62]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d', 
                       h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scaled_dot = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attention = F.softmax(scaled_dot, dim=-1)
        attention = self.att_drop(attention)
        
        x = torch.matmul(attention, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.projection(x)
        return x, attention


# In[63]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, attention = self.attn(x_norm)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attention


# In[64]:


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)
        attention_maps = []
        
        for block in self.transformer:
            x, attention = block(x)
            attention_maps.append(attention)
            
        x = self.norm(x)
        cls_token = x[:, 0]
        output = self.head(cls_token)
        
        if return_attention:
            return output, attention_maps
        return output


# In[65]:


class HemorrhageDataset(Dataset):
    def __init__(self, root_dir, transform=None, silent=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular', 
                               'subarachnoid', 'subdural', 'any']
        
        # Initialize patient_folders here
        self.patient_folders = [f for f in self.root_dir.iterdir() if f.is_dir()] if root_dir else []
        
        if root_dir and not silent:  # Only load dataset if root_dir is provided and not silent
            print("Loading dataset...")
            self._load_dataset()
            print(f"Found {len(self.file_paths)} valid samples")

    def _load_dataset(self):
        for patient_folder in self.patient_folders:
            # Find the NIFTI file
            nifti_files = list(patient_folder.glob(f"{patient_folder.name}_*.nii.gz"))
            if not nifti_files:
                continue
            
            # Get the labels file
            labels_file = patient_folder / "hemorrhage_labels.csv"
            if not labels_file.exists():
                continue
                
            # Read labels
            labels_df = pd.read_csv(labels_file)
            
            # Extract labels for all hemorrhage types
            try:
                labels = [int(labels_df[labels_df['hemorrhage_type'] == htype]['label'].values[0])
                         for htype in self.hemorrhage_types]
                
                # Store file path and labels
                self.file_paths.append(nifti_files[0])
                self.labels.append(labels)
            except Exception as e:
                print(f"Error processing {patient_folder}: {e}")
                continue
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load NIFTI file
        nifti_path = self.file_paths[idx]
        nifti_img = nib.load(nifti_path)
        image = nifti_img.get_fdata()
        
        # Take middle slice if 3D
        if len(image.shape) == 3:
            middle_slice = image.shape[2] // 2
            image = image[:, :, middle_slice]
        
        # Normalize the image
        image = self.normalize_image(image)
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Add channel dimension
        image = image.unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, labels

    def normalize_image(self, image):
        """Normalize image values to [0, 1] range"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val != min_val:
            image = (image - min_val) / (max_val - min_val)
        return image


# In[71]:


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def save_activation(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output
        
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)
    
    def generate_cam(self, input_image, target_class):
        b, c, h, w = input_image.shape
        
        # Forward pass
        self.model.zero_grad()
        output, attention_maps = self.model(input_image, return_attention=True)
        
        # Target for backprop
        if target_class is None:
            target_class = output.argmax(dim=1)
            
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        with torch.no_grad():
            # Get gradients and activations
            gradients = self.gradients[0]  # [num_tokens, embed_dim]
            activations = self.activations[0]  # [num_tokens, embed_dim]
            
            # Remove CLS token
            gradients = gradients[1:]  # [num_patches, embed_dim]
            activations = activations[1:]  # [num_patches, embed_dim]
            
            # Calculate importance weights
            weights = gradients.mean(dim=0)  # [embed_dim]
            
            # Weighted combination of activation maps
            cam = torch.matmul(activations, weights)  # [num_patches]
            
            # Reshape to square grid
            num_patches = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(num_patches, num_patches)
            
            # Normalize
            cam = F.relu(cam)
            if torch.max(cam) != 0:  # Avoid division by zero
                cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
            
            # Resize to input resolution
            cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            cam = cam.squeeze()
            
        return cam, attention_maps


# In[67]:


# Modify ViT for multi-label classification
class HemorrhageViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=1, 
                 num_hemorrhage_types=6, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=embed_dim,  # Use embed_dim as intermediate dimension
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        # Add multi-label classification head
        self.multilabel_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_hemorrhage_types),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            features, attention_maps = self.vit(x, return_attention=True)
            output = self.multilabel_head(features)
            return output, attention_maps
        else:
            features = self.vit(x)
            output = self.multilabel_head(features)
            return output
        
def save_training_metrics(batch_losses, epoch_losses, timestamp=None):
    """Save training metrics to CSV files"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save batch losses
    batch_df = pd.DataFrame({'batch_number': range(len(batch_losses)), 'loss': batch_losses})
    batch_df.to_csv(f'batch_losses_{timestamp}.csv', index=False)
    
    # Save epoch losses
    epoch_df = pd.DataFrame({'epoch_number': range(len(epoch_losses)), 'loss': epoch_losses})
    epoch_df.to_csv(f'epoch_losses_{timestamp}.csv', index=False)
    
    return timestamp


def plot_training_metrics(csv_path, metric_type='batch', save_path=None, display=True):
    """Generate publication-quality training metric plots"""
    print(f"Reading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} {metric_type} records")
    
    plt.figure(figsize=(12, 6))
    
    # Create the plot
    plt.plot(df[f'{metric_type}_number'], df['loss'], 
            color='#2671b8', linewidth=2, 
            marker='o' if metric_type == 'epoch' else None)
    
    # Customize the plot
    plt.title(f'{metric_type.capitalize()} Loss During Training', 
             fontsize=14, pad=20)
    plt.xlabel(f'{metric_type.capitalize()} Number', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f'{metric_type}_losses_plot_publication.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    
    # Display in notebook if requested
    if display:
        plt.show()
    else:
        plt.close()

        
def train_hemorrhage_model(root_dir, num_epochs=10, batch_size=8):
    # Set up GPU devices and metrics tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_losses = []
    epoch_losses = []
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs!")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = HemorrhageDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size * num_gpus if torch.cuda.is_available() else batch_size, 
        shuffle=True, 
        num_workers=4 * num_gpus if torch.cuda.is_available() else 4
    )
    
    # Initialize model
    model = HemorrhageViT(
        image_size=224,
        patch_size=16,
        in_channels=1,
        num_hemorrhage_types=6
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    total_batches = len(dataloader)
    print(f"Total batches per epoch: {total_batches}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        
        pbar = tqdm(enumerate(dataloader), total=total_batches, 
                   desc=f"Epoch {epoch+1}/{num_epochs}",
                   bar_format='{l_bar}{bar:30}{r_bar}')
        
        for batch_idx, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Loss: {batch_loss:.4f}")
            print(f"Batch [{batch_idx+1}/{total_batches}] Loss: {batch_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / total_batches
        epoch_losses.append(avg_epoch_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'batch_losses': batch_losses,
            'epoch_losses': epoch_losses
        }
        torch.save(checkpoint, f'hemorrhage_model_epoch_{epoch+1}.pth')
    
    # Save metrics
    save_training_metrics(batch_losses, epoch_losses, timestamp)
    
    print("\nTraining completed!")
    return model.eval(), timestamp  # Return both model and timestamp


def analyze_hemorrhage_with_gradcam(model, nifti_path):
    """Analyze a CT scan for hemorrhages and visualize with Grad-CAM"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the image
    nifti_img = nib.load(nifti_path)
    image = nifti_img.get_fdata()
    
    # Take middle slice if 3D
    if len(image.shape) == 3:
        middle_slice = image.shape[2] // 2
        image = image[:, :, middle_slice]
    
    # Create temporary dataset instance for normalization
    temp_dataset = HemorrhageDataset(root_dir="", silent=True)
    image = temp_dataset.normalize_image(image)
    
    # Prepare image tensor
    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
    image = image.to(device)
    
    # Initialize Grad-CAM
    target_layer = model.module.vit.transformer[-1] if hasattr(model, 'module') else model.vit.transformer[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Get predictions and CAM for each hemorrhage type
    hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular', 
                       'subarachnoid', 'subdural', 'any']
    results = []
    
    with torch.no_grad():
        predictions = model(image)
    
    for idx, h_type in enumerate(hemorrhage_types):
        cam, attention_maps = grad_cam.generate_cam(image, target_class=idx)
        results.append({
            'type': h_type,
            'probability': predictions[0, idx].item(),
            'cam': cam,
            'attention_maps': attention_maps
        })
    
    return results, image

def visualize_gradcam_results(image, results, save_dir='gradcam_results', display=True):
    """
    Visualize and save Grad-CAM results for each detected hemorrhage type
    """
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert image to numpy for visualization
    orig_image = image.squeeze().cpu().numpy()
    
    if not results:
        print("No hemorrhages detected above threshold. Displaying all predictions anyway.")
    
    for result in results:
        h_type = result['type']
        prob = result['probability']
        cam = result['cam'].squeeze().cpu().numpy()
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Original image
        ax1.imshow(orig_image, cmap='gray')
        ax1.set_title('Original CT Scan')
        ax1.axis('off')
        
        # Grad-CAM heatmap
        ax2.imshow(orig_image, cmap='gray')
        heatmap = ax2.imshow(cam, cmap='jet', alpha=0.5)
        ax2.set_title(f'{h_type} Hemorrhage\nProbability: {prob:.3f}')
        ax2.axis('off')
        plt.colorbar(heatmap, ax=ax2)
        
        # Attention map
        if result['attention_maps']:
            att_map = result['attention_maps'][-1][0, 0].mean(0).cpu().numpy()
            att_size = int(np.sqrt(att_map.shape[0] - 1))
            att_map = att_map[1:].reshape(att_size, att_size)
            ax3.imshow(att_map, cmap='viridis')
            ax3.set_title('Attention Map')
            ax3.axis('off')
            plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax3)
        
        plt.suptitle(f'Grad-CAM Visualization for {h_type} Hemorrhage', y=1.05)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f'gradcam_{h_type.lower()}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Display in notebook if requested
        if display:
            plt.show()
        else:
            plt.close()
        
        print(f"Saved Grad-CAM visualization for {h_type} to {save_path}")


# In[72]:


if __name__ == "__main__":
    ROOT_DIR = "/projectnb/ec523kb/projects/hemorrhage-classification/stage_2_train_sorted_nifti_pruned"
    
    print("Starting model training...")
    model, training_timestamp = train_hemorrhage_model(
        root_dir=ROOT_DIR,
        num_epochs=1,
        batch_size=8
    )
    
    # Save the trained model
    torch.save(model.state_dict(), 'hemorrhage_model.pth')
    print("Model saved to hemorrhage_model.pth")
    
    # Plot the training metrics
    print("\nPlotting training metrics...")
    plot_training_metrics(f'batch_losses_{training_timestamp}.csv', metric_type='batch', display=True)
    plot_training_metrics(f'epoch_losses_{training_timestamp}.csv', metric_type='epoch', display=True)
    
    # Test and visualize a specific case
    print("\nAnalyzing a specific case...")
    test_file = "/projectnb/ec523kb/projects/hemorrhage-classification/stage_2_train_sorted_nifti_pruned/ID_0000298a7d/ID_0000298a7d.nii.gz"
    
    results, image = analyze_hemorrhage_with_gradcam(model, test_file)
    
    # Visualize all results
    visualize_gradcam_results(image, results, display=True)


# In[ ]:




