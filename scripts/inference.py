from huggingface_hub import snapshot_download
from config.vit_config import ViTConfig
from models.vit_model import MedicalViTModel
import json
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern
import json
from tqdm import tqdm
import io

# Set your Hugging Face token here
hf_token = 'HF_TOKEN'

# Download the private repo
snapshot_download(
    repo_id="Gobi2005/medical-vit-cui-classifier",
    token=hf_token,
    local_dir="model"
)


# Load config with correct num_labels (MUST match the model that saved the weights)
with open("model/config.json", "r") as f:
    cfg_dict = json.load(f)

# This config should say "num_labels": 19
cfg_dict["num_labels"] = 19


# Build model
config = ViTConfig(**cfg_dict)
model = MedicalViTModel(config)

# Load weights safely
state_dict = load_file("model/model.safetensors")
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print("Model loaded.")
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CUI class names
with open("model/cui_classes.json", "r") as f:
    cui_classes = json.load(f)["classes"]

cui_to_index = {c: i for i, c in enumerate(cui_classes)}
print(cui_to_index)

class ParquetImageDataset(Dataset):
    def __init__(self, parquet_path, transform_size=(224, 224)):
        self.df = pd.read_parquet(parquet_path)
        self.size = transform_size
        self.mean = [0.485, 0.456, 0.406, 0.5]
        self.std = [0.229, 0.224, 0.225, 0.25]

    def __len__(self):
        return len(self.df)

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(image)
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        lbp = local_binary_pattern(gray_uint8, P=24, R=3, method='uniform')

        if lbp.max() > lbp.min():
            lbp_norm = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
        else:
            lbp_norm = np.zeros_like(lbp, dtype=np.uint8)

        img_4ch = np.dstack([img_array, lbp_norm])
        img_4ch = Image.fromarray(img_4ch.astype(np.uint8)).resize(self.size)

        img_tensor = torch.from_numpy(np.array(img_4ch)).permute(2, 0, 1).float() / 255.0
        for i in range(4):
            img_tensor[i] = (img_tensor[i] - self.mean[i]) / self.std[i]
        return img_tensor

    def __getitem__(self, idx):
        image_bytes = self.df.loc[idx, 'image']['bytes']
        img_tensor = self.preprocess_image(image_bytes)
        
        # Parse labels
        try:
            raw = self.df.loc[idx, 'cui']
            labels = eval(raw) if isinstance(raw, str) else raw
            label_vec = torch.zeros(len(cui_classes))
            for cui in labels:
                if cui in cui_to_index:
                    label_vec[cui_to_index[cui]] = 1
        except:
            label_vec = torch.zeros(len(cui_classes))

        return img_tensor, label_vec, idx


parquet_path = "path_to_parquet"
dataset = ParquetImageDataset(parquet_path)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

# Inference on all images (batched with tqdm)
all_preds, all_labels, all_indices = [], [], []

with torch.no_grad():
    for images, labels, indices in tqdm(dataloader, desc="Running Inference", total=len(dataloader)):
        images = images.to(device)
        outputs = model(pixel_values=images)
        probs = torch.sigmoid(outputs["logits"]) > 0.5
        all_preds.append(probs.cpu())
        all_labels.append(labels)
        all_indices.extend(indices.tolist())


all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Evaluation
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Save Predictions
df = pd.read_parquet(parquet_path)
predictions = []

for i, idx in enumerate(all_indices):
    pred_labels = [cui_classes[j] for j in np.where(all_preds[i] == 1)[0]]
    true_labels = [cui_classes[j] for j in np.where(all_labels[i] == 1)[0]]
    predictions.append({
        "index": int(idx),
        "predicted_cuis": pred_labels,
        "true_cuis": true_labels
    })

# Save to JSON
with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

# Save to CSV
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("predictions.csv", index=False)

print("Saved predictions to predictions.json and predictions.csv")

# Per-class metrics
per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

metrics_df = pd.DataFrame({
    "CUI": cui_classes,
    "Precision": per_class_precision,
    "Recall": per_class_recall,
    "F1-Score": per_class_f1
})
metrics_df.to_csv("per_class_metrics.csv", index=False)
print("Saved per-class metrics to per_class_metrics.csv")


mcm = multilabel_confusion_matrix(all_labels, all_preds)
conf_matrix_summary = []

for i, cui in enumerate(cui_classes):
    tn, fp, fn, tp = mcm[i].ravel()
    conf_matrix_summary.append({
        "CUI": cui,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    })

conf_df = pd.DataFrame(conf_matrix_summary)
conf_df.to_csv("confusion_matrix_per_class.csv", index=False)
print("Saved multilabel confusion matrix to confusion_matrix_per_class.csv")


# Reload parquet for access to image bytes
df = pd.read_parquet(parquet_path)

for i in range(5):  # Show first 5 images
    idx = all_indices[i]
    row = df.iloc[idx]
    image_bytes = row['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    pred_labels = [cui_classes[j] for j in np.where(all_preds[i] == 1)[0]]
    true_labels = [cui_classes[j] for j in np.where(all_labels[i] == 1)[0]]

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"True: {', '.join(true_labels)}\nPred: {', '.join(pred_labels)}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"prediction_vis_{i}.png")  # Saves each image
    plt.show()