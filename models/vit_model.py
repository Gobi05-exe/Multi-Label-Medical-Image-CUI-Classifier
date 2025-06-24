
import timm
import torch.nn as nn
from transformers import PreTrainedModel
from config.vit_config import ViTConfig
import torch
from typing import Dict, Optional

class MedicalViTModel(PreTrainedModel):
    """Modified ViT model for 4-channel input and multi-label classification."""
    
    config_class = ViTConfig
    
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config
        
        # Load pretrained ViT
        self.vit = timm.create_model(
            config.model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Modify first conv layer for 4 channels
        original_conv = self.vit.patch_embed.proj
        self.vit.patch_embed.proj = nn.Conv2d(
            4,  # 4 input channels (RGB + LBP)
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize new weights
        with torch.no_grad():
            # Copy RGB weights and average them for the 4th channel
            rgb_weights = original_conv.weight.data
            self.vit.patch_embed.proj.weight.data[:, :3, :, :] = rgb_weights
            self.vit.patch_embed.proj.weight.data[:, 3:4, :, :] = rgb_weights.mean(dim=1, keepdim=True)
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_labels)
        )
        
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract features using ViT
        features = self.vit(pixel_values)
        
        # Classification
        logits = self.classifier(features)
        
        output = {"logits": logits}
        
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            output["loss"] = loss
            
        return output