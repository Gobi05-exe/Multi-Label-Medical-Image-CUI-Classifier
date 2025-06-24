
# Multi-Label Medical Image CUI Classifier

This project implements a Vision Transformer (ViT) model for **multi-label classification of medical images**, incorporating **4-channel inputs (RGB + LBP)** and trained for **Concept Unique Identifier (CUI)** detection.

The project uses:
- Vision Transformer (ViT) backbone with 4-channel support
- Local Binary Patterns (LBP) as an additional feature channel
- Hugging Face integration for dataset management and model hosting
- Modular, maintainable Python structure

**Note:** The project includes Jupyter Notebooks for interactive training and inference, located in the notebooks/ directory (notebooks/train.ipynb and notebooks/inference.ipynb) to facilitate experimentation and result visualization. 

## Project Structure

```
medical_vit_project/
├── config/               # Model configuration
├── data/                 # Datasets and image transforms
├── models/               # Model architecture
├── trainers/             # Training loop
├── utils/                # Logging, seeding, Hugging Face utilities
├── scripts/              # Training and inference entry points
├── outputs/              # Saved checkpoints and results
├── requirements.txt      # Project dependencies
├── .gitignore
└── README.md
```

## Setup

1. **Clone the Repository**
```bash
git clone <your_repo_url>
cd medical_vit_project
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Hugging Face Authentication**
```bash
huggingface-cli login
```
You must have an active Hugging Face account and token.

## Training the Model

```bash
python scripts/train.py
```

Training pipeline includes:
- Automatic dataset download via Hugging Face `datasets`
- Model training with early stopping and checkpoints
- Model upload to Hugging Face Hub after completion

**Modify `scripts/train.py` to set:**
- Hugging Face token
- Dataset repo ID
- Number of epochs
- Other hyperparameters

## Inference on New Data

1. Place your trained model and config under `model/`
2. Modify `scripts/inference.py` to set:
    - Model path
    - Input data path
    - Hugging Face token

3. Run:

```bash
python scripts/inference.py
```

This script:
- Loads the trained ViT model
- Runs batch inference on test parquet files
- Saves predictions as `.csv`

## Features

✅ Vision Transformer Backbone  
✅ 4-Channel Input with LBP Integration  
✅ Multi-Label Classification for CUIs  
✅ PyTorch Lightning-like Trainer  
✅ GradCAM Visualizations  
✅ Hugging Face Dataset & Model Hub Support  
✅ Reproducible Seeds and Logging  

## Dependencies

See `requirements.txt` for detailed dependencies including:
- PyTorch
- timm
- transformers
- huggingface_hub
- scikit-learn
- scikit-image
- matplotlib

## Notes

- LBP computation uses `skimage.feature.local_binary_pattern`
- Ensure CUI class labels match across training and inference

## Example Use Case

This setup is ideal for:
- Radiology image analysis
- Medical concept extraction

## Sample inference results

![Screenshot from 2025-06-24 22-43-00](https://github.com/user-attachments/assets/842abf20-2377-487e-ab7b-803dc3bddc8b)

![Screenshot from 2025-06-24 22-43-13](https://github.com/user-attachments/assets/fe2921d9-7a05-42e2-a1da-0bf3cdd27204)

![Screenshot from 2025-06-24 22-43-33](https://github.com/user-attachments/assets/cb89e756-2319-48ee-82b3-5c650c3937ec)

## License

Released under the **Apache 2.0 License**

## Acknowledgements

Built using:
- [Hugging Face Transformers](https://huggingface.co/)
- [timm Vision Library](https://github.com/rwightman/pytorch-image-models)
- [PyTorch](https://pytorch.org/)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
