import torch
from typing import Tuple, Optional
import logging
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from data.dataset import MedicalImageDataset
from data.transforms import FourChannelTransform
from seed_utils import set_seeds



def create_data_loaders(
    train_parquet: str,
    val_parquet: str,
    batch_size: int = 16,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[DataLoader, DataLoader, MultiLabelBinarizer]:
    """Create training and validation data loaders."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set seeds if provided
    if seed is not None:
        set_seeds(seed)
        logger.info(f"Seeds set to {seed} for reproducibility")
    
    # Transform
    transform = FourChannelTransform(size=(224, 224))
    
    # Datasets
    train_dataset = MedicalImageDataset(
        train_parquet, 
        transform=transform, 
        is_training=True,
        device=device,
        logger=logger
    )
    val_dataset = MedicalImageDataset(
        val_parquet, 
        transform=transform, 
        is_training=False,
        device=device,
        logger=logger
    )
    
    # Share the multilabel binarizer
    val_dataset.set_mlb(train_dataset.mlb)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset.mlb