"""Checkpoint management utilities."""
import os
import torch
import json
import datetime
from typing import Dict, Any, Optional, List
import shutil


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Track saved checkpoints
        self.checkpoints = []
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       accuracy: float,
                       metrics: Optional[Dict[str, Any]] = None,
                       is_best_loss: bool = False,
                       is_best_acc: bool = False,
                       model_config: Optional[Dict[str, Any]] = None) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            accuracy: Current accuracy
            metrics: Additional metrics
            is_best_loss: Whether this is the best loss checkpoint
            is_best_acc: Whether this is the best accuracy checkpoint
            model_config: Model configuration dictionary
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': timestamp,
            'metrics': metrics or {},
            'model_config': model_config or {}
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch:04d}_{timestamp}.pth'
        )
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoints.append((checkpoint_path, epoch, loss, accuracy))
        
        # Save best loss model
        if is_best_loss:
            best_loss_path = os.path.join(self.checkpoint_dir, 'best_loss.pth')
            torch.save(checkpoint_data, best_loss_path)
            self.best_loss = loss
        
        # Save best accuracy model
        if is_best_acc:
            best_acc_path = os.path.join(self.checkpoint_dir, 'best_accuracy.pth')
            torch.save(checkpoint_data, best_acc_path)
            self.best_accuracy = accuracy
        
        # Save latest model
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint_data, latest_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Checkpoint data dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"with loss {checkpoint['loss']:.4f} and accuracy {checkpoint['accuracy']:.4f}")
        
        return checkpoint
    
    def load_best_model(self, 
                       model: torch.nn.Module,
                       criterion: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """Load the best model checkpoint.
        
        Args:
            model: Model to load state into
            criterion: Either 'accuracy' or 'loss'
            
        Returns:
            Checkpoint data if found, None otherwise
        """
        if criterion == 'accuracy':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_accuracy.pth')
        elif criterion == 'loss':
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_loss.pth')
        else:
            raise ValueError("Criterion must be 'accuracy' or 'loss'")
        
        if os.path.exists(checkpoint_path):
            return self.load_checkpoint(checkpoint_path, model)
        else:
            print(f"Best {criterion} checkpoint not found")
            return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if not found
        """
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        return latest_path if os.path.exists(latest_path) else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoint_info = []
        
        for checkpoint_path, epoch, loss, accuracy in self.checkpoints:
            if os.path.exists(checkpoint_path):
                checkpoint_info.append({
                    'path': checkpoint_path,
                    'epoch': epoch,
                    'loss': loss,
                    'accuracy': accuracy,
                    'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
                })
        
        return checkpoint_info
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch and remove oldest
            self.checkpoints.sort(key=lambda x: x[1])  # Sort by epoch
            
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint[0]):
                    os.remove(old_checkpoint[0])
                    print(f"Removed old checkpoint: {old_checkpoint[0]}")
    
    def _save_checkpoint_info(self) -> None:
        """Save checkpoint information to JSON file."""
        info = {
            'checkpoints': self.list_checkpoints(),
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'max_checkpoints': self.max_checkpoints
        }
        
        info_path = os.path.join(self.checkpoint_dir, 'checkpoint_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def delete_all_checkpoints(self) -> None:
        """Delete all checkpoints and reset manager."""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoints = []
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        print("All checkpoints deleted")
    
    def get_checkpoint_size(self) -> float:
        """Get total size of all checkpoints in MB.
        
        Returns:
            Total size in MB
        """
        total_size = 0
        
        for checkpoint_path, _, _, _ in self.checkpoints:
            if os.path.exists(checkpoint_path):
                total_size += os.path.getsize(checkpoint_path)
        
        # Add special checkpoints
        for special_file in ['best_loss.pth', 'best_accuracy.pth', 'latest.pth']:
            special_path = os.path.join(self.checkpoint_dir, special_file)
            if os.path.exists(special_path):
                total_size += os.path.getsize(special_path)
        
        return total_size / (1024 * 1024)  # Convert to MB
