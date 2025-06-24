
from transformers import PretrainedConfig

class ViTConfig(PretrainedConfig):
    """Configuration class for ViT model."""
    
    model_type = "vit_medical_cui"
    
    def __init__(
        self,
        num_labels: int = 100,
        model_name: str = "vit_base_patch16_224",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.model_name = model_name

