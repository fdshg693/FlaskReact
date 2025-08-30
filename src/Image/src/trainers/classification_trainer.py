"""Classification trainer for wood classification."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import cv2

from .base_trainer import BaseTrainer
from ..utils.metrics import MetricsCalculator
from ..utils.visualization import Visualizer
from ..utils.checkpoint import CheckpointManager
from ..utils.logger import Logger


class ClassificationTrainer(BaseTrainer):
    """Trainer class for classification tasks."""
    
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 logger: Optional[Logger] = None,
                 visualizer: Optional[Visualizer] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 early_stopping_patience: Optional[int] = None,
                 grad_clip_norm: Optional[float] = None):
        """Initialize classification trainer.
        
        Args:
            model: Neural network model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to run training on
            scheduler: Learning rate scheduler (optional)
            logger: Logger instance (optional)
            visualizer: Visualizer instance (optional)
            checkpoint_manager: Checkpoint manager (optional)
            early_stopping_patience: Early stopping patience (optional)
            grad_clip_norm: Gradient clipping norm (optional)
        """
        super(ClassificationTrainer, self).__init__(model, device)
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.visualizer = visualizer
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip_norm = grad_clip_norm
        
        # Early stopping state
        self.early_stopping_counter = 0
        self.should_stop_early = False
        
        # Move criterion to device
        if hasattr(self.criterion, 'to'):
            self.criterion.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        
        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(self.model.num_class)
        
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate loss
            loss = self.criterion(output, target.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            metrics_calc.update(output, target, loss.item())
            total_samples += data.size(0)
        
        return metrics_calc.get_average_loss(), metrics_calc.get_accuracy()
    
    def validate_epoch(self, val_loader: DataLoader, 
                      show_sample: bool = False) -> Tuple[float, float, Dict[str, Any]]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            show_sample: Whether to show sample images
            
        Returns:
            Tuple of (average_loss, average_accuracy, detailed_metrics)
        """
        self.model.eval()
        
        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(self.model.num_class)
        
        sample_images = None
        sample_labels = None
        sample_predictions = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target.float())
                
                # Update metrics
                metrics_calc.update(output, target, loss.item())
                
                # Store first batch for visualization
                if batch_idx == 0 and show_sample:
                    sample_images = data.cpu()
                    sample_labels = target.cpu()
                    sample_predictions = output.cpu()
                    
                    # Show sample image using OpenCV (for compatibility)
                    if len(data) > 0:
                        sample_img = data[0].cpu().numpy().transpose(1, 2, 0)
                        if sample_img.max() <= 1.0:
                            sample_img = (sample_img * 255).astype('uint8')
                        cv2.imshow('sample', sample_img)
                        cv2.waitKey(1)
        
        # Get detailed metrics
        detailed_metrics = metrics_calc.get_detailed_metrics()
        
        # Save sample images if visualizer is available
        if (self.visualizer and sample_images is not None and 
            sample_labels is not None and sample_predictions is not None):
            self.visualizer.save_sample_images(
                sample_images, sample_labels, sample_predictions,
                save_name=f'validation_samples_epoch_{self.current_epoch}'
            )
        
        return metrics_calc.get_average_loss(), metrics_calc.get_accuracy(), detailed_metrics
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            dataset_for_augmentation: Optional[Any] = None) -> Dict[str, Any]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            dataset_for_augmentation: Dataset with augmentation control (optional)
            
        Returns:
            Training history dictionary
        """
        if self.logger:
            self.logger.log_info(f"Starting training for {epochs} epochs")
            self.logger.log_model_info(self.model.get_model_info())
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Enable augmentation for training
            if dataset_for_augmentation:
                dataset_for_augmentation.set_augmentation(True)
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Disable augmentation for validation
            if dataset_for_augmentation:
                dataset_for_augmentation.set_augmentation(False)
            
            # Validate epoch
            val_loss, val_acc, detailed_metrics = self.validate_epoch(
                val_loader, show_sample=(epoch == 0)
            )
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update training history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check for best models
            is_best_loss = val_loss < self.best_loss
            is_best_acc = val_acc > self.best_accuracy
            
            if is_best_loss:
                self.best_loss = val_loss
                self.early_stopping_counter = 0
            
            if is_best_acc:
                self.best_accuracy = val_acc
            
            self.best_losses.append(self.best_loss)
            self.best_accuracies.append(self.best_accuracy)
            
            # Save checkpoint
            if self.checkpoint_manager:
                # Get model configuration from the model
                model_config = {}
                if hasattr(self.model, 'get_model_info'):
                    model_info = self.model.get_model_info()
                    model_config.update(model_info)
                
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    accuracy=val_acc,
                    metrics=detailed_metrics,
                    is_best_loss=is_best_loss,
                    is_best_acc=is_best_acc,
                    model_config=model_config
                )
            
            # Create visualizations
            if self.visualizer:
                # Training curves
                plot_array = self.visualizer.plot_training_curves(
                    self.train_losses, self.val_losses,
                    self.train_accuracies, self.val_accuracies,
                    self.best_losses, self.best_accuracies
                )
                
                # Show training curves
                self.visualizer.numpy_to_cv2_display(plot_array, 'Training Progress', 1)
                
                # Confusion matrix
                if 'confusion_matrix' in detailed_metrics:
                    heatmap = self.visualizer.plot_confusion_matrix(
                        detailed_metrics['confusion_matrix'],
                        save_name=f'confusion_matrix_epoch_{epoch}'
                    )
                    
                    # Show confusion matrix
                    if is_best_loss:
                        cv2.imshow('confusion_matrix_best_loss', heatmap)
                    if is_best_acc:
                        cv2.imshow('confusion_matrix_best_acc', heatmap)
                    cv2.waitKey(1)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            if self.logger:
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    best_loss=self.best_loss,
                    best_acc=self.best_accuracy,
                    learning_rate=self.get_learning_rate(),
                    elapsed_time=epoch_time
                )
                
                # Log detailed metrics periodically
                if epoch % 100 == 0 or is_best_acc:
                    self.logger.log_metrics(detailed_metrics, f"Epoch {epoch} Validation")
            
            # Early stopping check
            if self.early_stopping_patience:
                if not is_best_loss:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        if self.logger:
                            self.logger.log_info(
                                f"Early stopping triggered after {self.early_stopping_patience} "
                                f"epochs without improvement"
                            )
                        self.should_stop_early = True
                        break
            
            # Print progress (compatible with original code)
            print(f'Epoch {epoch:4d} | '
                  f'Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | '
                  f'Best Loss: {self.best_loss:.6f} | Best Acc: {self.best_accuracy:.4f} | '
                  f'Time: {epoch_time:.2f}s')
        
        # Training completed
        total_time = time.time() - start_time
        
        if self.logger:
            # Save numpy logs for compatibility
            self.logger.save_numpy_logs(
                self.train_losses, self.train_accuracies,
                self.val_losses, self.val_accuracies,
                self.best_losses, self.best_accuracies
            )
            
            # Create summary
            self.logger.create_summary(
                final_train_acc=self.train_accuracies[-1] if self.train_accuracies else 0,
                final_val_acc=self.val_accuracies[-1] if self.val_accuracies else 0,
                best_val_acc=self.best_accuracy,
                total_epochs=self.current_epoch + 1,
                total_time=total_time
            )
        
        return self.get_training_history()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if self.logger:
            self.logger.log_info("Starting evaluation on test set")
        
        test_loss, test_acc, detailed_metrics = self.validate_epoch(test_loader)
        
        if self.logger:
            self.logger.log_info(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.4f}")
            self.logger.log_metrics(detailed_metrics, "Test Set")
        
        return detailed_metrics
