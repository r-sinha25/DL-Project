# Explainable Transformer-Based Classification and Segmentation of Intracranial Hemorrhages
Team members: 

- Rupali Sinha | rsinha@bu.edu
- Albert Zhao | albertz@bu.edu
- Ishan Bhattacharjee | ibhattac@bu.edu
- Benjamin Axline | baxline@bu.edu

[📄 View Proposal (PDF)](ACTUAL523%20Project%20Proposal%20Template.pdf)


--
## **Abstract**
Intracranial Hemorrhage (ICH), a life-threatening condition, is traditionally diagnosed through manual inspection of brain CT scans. To improve the accuracy and efficiency of detection, we developed two deep learning models:
1. A **pretrained segmentation model** (DeepBleed) to establish ground-truth segmentation masks.
2. A **Vision Transformer (ViT)** tailored for multi-label classification, incorporating Grad-CAM explainability for visualization.

The goal is to compare these models' performance on detecting **five hemorrhage subtypes** (Epidural, Subdural, Subarachnoid, Intraventricular, Intraparenchymal) and the general "any" label. Key evaluation metrics include **Accuracy**, **F1 Score**, **AUROC**, and **AUPRC**.

---

## **1. Data Installation**
### **Dataset**
The dataset is sourced from the [RSNA Intracranial Hemorrhage Detection competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). It contains DICOM brain CT scans with multi-label annotations.

### **Steps to Download the Dataset**
Run the following command to download the dataset using Kaggle CLI:

```bash
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
```

Unzip the downloaded dataset:
```bash
unzip rsna-intracranial-hemorrhage-detection.zip -d data/
```

---

## **2. Project Pipeline**

### **2.1 Data Preprocessing**
- Convert DICOM files to **NIfTI format** for efficient 3D processing.
- Normalize and augment the data with **random flips, rotations**, and **intensity scaling**.
- Stratified sampling ensures a balanced split between training and testing datasets.

### **2.2 Pretrained Segmentation Model**
- Use **DeepBleed** (3D CNN) to generate segmentation masks, establishing ground truth for hemorrhage regions.
- Process NIfTI CT images with a multi-GPU pipeline for fast inference.

### **2.3 Vision Transformer (ViT)**
- Fine-tuned ViT for **multi-label classification** of hemorrhage subtypes.
- Integrated **Grad-CAM explainability** to generate heatmaps highlighting critical regions for model predictions.
- Optimized training using **Focal Loss** to handle class imbalance.

### **2.4 Training Strategy**
- **Optimizer**: Adam optimizer with cyclic learning rate scheduling.
- **Loss Function**: Weighted **Focal Loss** for multi-label classification.
- **Regularization**: Early stopping and gradient clipping to stabilize training.

### **2.5 Evaluation Metrics**
- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Combines precision and recall to evaluate imbalanced classes.
- **AUROC**: Ability to distinguish between hemorrhage presence and absence.
- **AUPRC**: Precision-Recall curve performance for rare hemorrhage types.

---

## **3. Installation and Requirements**
### **3.1 Prerequisites**
- **Python** (>=3.8)
- **PyTorch** (>=1.8)
- **MONAI** (Medical Open Network for AI)
- **TensorFlow** (for DeepBleed)
- **CUDA** (for GPU acceleration)
- **scikit-learn**
- **SimpleITK**
- **NumPy**
- **Matplotlib**

### **3.2 Installation Steps**
1. Clone this repository:
   ```bash
   git clone https://github.com/r-sinha25/DL-Project.git
   cd DL-Project
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate    # For Windows
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **4. Running the Models**
### **4.1 Ground Truth Segmentation (DeepBleed)**
Generate segmentation masks using DeepBleed:
```bash
python run_deepbleed_segmentation.py --input_dir data/nifti --output_dir results/segmentation
```

### **4.2 Multi-Label Classification (ViT)**
Train the Vision Transformer for multi-label hemorrhage classification:
```bash
python train_vit_classifier.py --config configs/multi_label_vit.yaml
```

### **4.3 Grad-CAM Explainability**
Generate Grad-CAM heatmaps for model interpretability:
```bash
python gradcam_vit.py --input_dir data/nifti --model_checkpoint results/checkpoints/vit.pth --output_dir results/grad_cam
```

---

## **5. Results**
The performance of our Vision Transformer model is summarized below:

| **Metric**          | **Training Set** | **Testing Set** |
|----------------------|------------------|-----------------|
| **Accuracy (%)**     | 86.5            | 84.3           |
| **F1 Score (Weighted)** | 0.79           | 0.75           |
| **AUROC**            | 0.82            | 0.78           |
| **AUPRC**            | 0.68            | 0.63           |

- The model achieved a testing accuracy of **84.3%** and an F1 Score of **0.75**, demonstrating its reliability in handling multiple hemorrhage types.
- Grad-CAM outputs provided visual explainability, highlighting regions of the brain critical for decision-making.

---

## **6. Directory Structure**
```
DL-Project/
│-- data/                 # Dataset folder
│   │-- dicom/            # Original DICOM files
│   │-- nifti/            # Converted NIfTI files
│-- results/              # Output directory
│   │-- segmentation/     # Ground truth segmentation masks
│   │-- grad_cam/         # Grad-CAM heatmaps
│   │-- metrics.json      # Evaluation metrics
│-- configs/              # Configuration files
│-- scripts/              # Preprocessing scripts
│-- train_vit_classifier.py   # Training script for Vision Transformer
│-- run_deepbleed_segmentation.py # Script for DeepBleed model
│-- gradcam_vit.py        # Grad-CAM generation script
│-- requirements.txt      # Required dependencies
```

---

## **7. Contributions**
| **Team Member**      | **Contributions**                                         |
|-----------------------|----------------------------------------------------------|
| **Rupali Sinha**      | Preprocessing, classification, model integration.       |
| **Albert Zhao**       | ViT model development, Grad-CAM explainability.         |
| **Ishan Bhattacharjee** | Pretrained segmentation, dataset setup.                |
| **Benjamin Axline**   | ViT evaluation, code optimization, model comparison.    |

---

## **8. Acknowledgments**
We extend our gratitude to the MONAI team and the creators of the DeepBleed model for providing open-source tools that enabled this project.

---

## **9. References**
- Liu, Q. et al. Project-MONAI, 2024.  
- Grewal, M. et al. RADnet: Radiologist-level accuracy for hemorrhage detection. IEEE, 2018.
- Wang et al. Dual-task Vision Transformer for ICH Classification. arXiv, 2024.



