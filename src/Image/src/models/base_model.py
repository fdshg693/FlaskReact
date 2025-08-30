"""Base model interface for wood classification models."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""
    
    def __init__(self, num_class: int, **kwargs):
        """Initialize base model.
        
        Args:
            num_class: Number of classification classes
            **kwargs: Additional model-specific parameters
        """
        super(BaseModel, self).__init__()
        self.num_class = num_class
        self.model_params = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model parameters and info
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'num_class': self.num_class,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_params': self.model_params
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model state dict.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, filepath)
    
    def load_model(self, filepath: str, strict: bool = True) -> None:
        """Load model state dict.
        
        Args:
            filepath: Path to load the model from
            strict: Whether to strictly enforce that the keys match
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'], strict=strict)
