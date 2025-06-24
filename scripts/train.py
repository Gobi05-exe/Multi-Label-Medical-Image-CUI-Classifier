from config.vit_config import ViTConfig
from models.vit_model import MedicalViTModel
from trainers.trainer import MedicalImageTrainer
from utils.logging_utils import setup_logging
from utils.seed_utils import set_seeds
from utils.hf_utils import push_to_huggingface, load_and_save_dataset
from huggingface_hub import login
from utils.load_data import create_data_loaders

import torch
from torch.utils.data import DataLoader

def main():
    login(token='YOUR_HF_TOKEN')  

       #get parquet files from HuggingFace
    load_and_save_dataset(train_pct=0.30,valid_pct=0.30)

    # Setup logging
    logger = setup_logging(level='INFO', log_file='training.log')
    
    # Configuration
    config = {
        'train_parquet': '/kaggle/working/dataset/train/train_dataset.parquet',
        'val_parquet': '/kaggle/working/dataset/valid/valid_dataset.parquet',
        'batch_size': 16,
        'num_epochs': 15,
        'repo_name': 'Gobi2005/medical-vit-cui-classifier',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("Starting medical image CUI classification pipeline")
    logger.info(f"Configuration: {config}")
    
    logger.info("Creating data loaders...")
    train_loader, val_loader, mlb = create_data_loaders(
        config['train_parquet'],
        config['val_parquet'],
        config['batch_size'],
        config['device'],
        config['seed'],
        logger
    )
    
    logger.info(f"Number of CUI classes: {len(mlb.classes_)}")
    
    # Create model
    model_config = ViTConfig(num_labels=len(mlb.classes_))
    model = MedicalViTModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params} total parameters ({trainable_params} trainable)")
    
    # Create trainer
    trainer = MedicalImageTrainer(
        model, 
        train_loader, 
        val_loader, 
        config['device'],
        logger
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(config['num_epochs'])
    
    # Get best metrics
    best_metrics = max(trainer.val_metrics, key=lambda x: x['f1_score'])
    logger.info(f"\nBest validation metrics:")
    logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
    logger.info(f"Precision: {best_metrics['precision']:.4f}")
    logger.info(f"Recall: {best_metrics['recall']:.4f}")
    
    # Prepare for Hugging Face upload
    model_card_data = {
        'base_model': 'vit_base_patch16_224',
        'num_classes': len(mlb.classes_),
        'best_f1': f"{best_metrics['f1_score']:.4f}",
        'best_precision': f"{best_metrics['precision']:.4f}",
        'best_recall': f"{best_metrics['recall']:.4f}"
    }
    
    # Push to Hugging Face
    logger.info("Pushing model to Hugging Face...")
    push_to_huggingface(
        model=model,
        mlb=mlb,
        repo_name=config['repo_name'],
        model_card_data=model_card_data,
        logger=logger
    )
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()