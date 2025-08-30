"""Base trainer interface for machine learning models."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import time


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 **kwargs):
        """Initialize base trainer.
        
        Args:
            model: Neural network model
            device: Device to run training on
            **kwargs: Additional trainer-specific parameters
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_losses = []
        self.best_accuracies = []
        
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        pass
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        pass
    
    @abstractmethod
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int) -> Dict[str, Any]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        pass
    
    def get_learning_rate(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate
        """
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history.
        
        Returns:
            Dictionary containing training history
        """
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_losses': self.best_losses,
            'best_accuracies': self.best_accuracies,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy
        }
    
    def reset_training_state(self) -> None:
        """Reset training state for fresh start."""
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_losses = []
        self.best_accuracies = []
