import os
import json
from huggingface_hub import HfApi
from datasets import load_dataset
from models.vit_model import MedicalViTModel
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from typing import Dict, Any, Optional

def push_to_huggingface(
    model: MedicalViTModel,
    mlb: MultiLabelBinarizer,
    repo_name: str,
    model_card_data: Dict[str, Any],
    local_dir: str = "./medical_vit_model",
    logger: Optional[logging.Logger] = None
) -> None:
    """Push model to Hugging Face Hub."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(local_dir)
    logger.info(f"Model saved to {local_dir}")
    
    # Save multilabel binarizer
    mlb_classes = mlb.classes_.tolist()
    with open(os.path.join(local_dir, "cui_classes.json"), "w") as f:
        json.dump({"classes": mlb_classes}, f, indent=2)
    
    # Create model card
    model_card = f"""---
tags:
- medical-imaging
- ViT
- LBP
- multi-label-classification
- CUIs
- vision-transformer
- pytorch
- medical-ai
language:
- en
library_name: transformers
pipeline_tag: image-classification
datasets:
- medical-cui-dataset
metrics:
- precision
- recall
- f1
---

# Medical Image CUI Classification Model

## Model Description

This model is a Vision Transformer (ViT) fine-tuned for multi-label classification of medical images with Concept Unique Identifiers (CUIs). The model uses a 4-channel input (RGB + Local Binary Pattern) for enhanced feature extraction from medical images.

## Model Architecture

- **Base Model**: {model_card_data.get('base_model', 'vit_base_patch16_224')}
- **Input Channels**: 4 (RGB + LBP features)
- **Input Size**: 224x224
- **Number of CUI Classes**: {model_card_data.get('num_classes', 'N/A')}
- **Classification Type**: Multi-label

## Training Details

- **Optimizer**: AdamW (lr=3e-5, weight_decay=0.01)
- **Loss Function**: BCEWithLogitsLoss
- **Scheduler**: CosineAnnealingLR
- **Best F1-Score**: {model_card_data.get('best_f1', 'N/A')}
- **Best Precision**: {model_card_data.get('best_precision', 'N/A')}
- **Best Recall**: {model_card_data.get('best_recall', 'N/A')}

## Usage

```python
from transformers import AutoModel, AutoConfig
import torch
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern

# Load model
config = AutoConfig.from_pretrained("{repo_name}")
model = AutoModel.from_pretrained("{repo_name}")

# Preprocess image (implement FourChannelTransform)
# ... your preprocessing code ...

# Inference
with torch.no_grad():
    outputs = model(pixel_values=processed_image)
    predictions = torch.sigmoid(outputs.logits) > 0.5
```

## License

This model is released under the Apache 2.0 License.
"""
    
    with open(os.path.join(local_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    # Create repository and upload
    try:
        api = HfApi()
        api.upload_folder(
        folder_path="/kaggle/working/medical_vit_model",
        repo_id="Gobi2005/medical-vit-cui-classifier",
)
        logger.info(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")


def load_and_save_dataset(
    repo_id: str = "eltorio/ROCOv2-radiology",
    train_pct: float = 0.10,
    valid_pct: float = 0.10,
    save_dir: str = "dataset"
):
    train_split = f"train[:{int(train_pct * 100)}%]"
    valid_split = f"validation[:{int(valid_pct * 100)}%]"

    # Create directories
    train_dir = os.path.join(save_dir, "train")
    valid_dir = os.path.join(save_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    print(f"Loading {int(train_pct * 100)}% of training data...")
    train_ds = load_dataset(repo_id, split=train_split)
    train_save_path = os.path.join(train_dir, "train_dataset.parquet")
    train_ds.to_parquet(train_save_path)
    print(f"Saved training dataset to {train_save_path}")

    print(f"Loading {int(valid_pct * 100)}% of validation data...")
    valid_ds = load_dataset(repo_id, split=valid_split)
    valid_save_path = os.path.join(valid_dir, "valid_dataset.parquet")
    valid_ds.to_parquet(valid_save_path)
    print(f"Saved validation dataset to {valid_save_path}")

    print("✅ Dataset loading and saving complete.")
    return train_ds, valid_ds