"""Main training script for wood classification."""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.wood_net import WoodNet
from src.datasets.wood_dataset import WoodDataset
from src.trainers.classification_trainer import ClassificationTrainer
from src.utils.logger import Logger
from src.utils.visualization import Visualizer
from src.utils.checkpoint import CheckpointManager
from config.base_config import ConfigManager


def main():
    """Main training function."""
    # Initialize configuration
    yaml_config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'
    config_manager = ConfigManager(config_path=yaml_config_path)
    config = config_manager.get_config()
    
    print(f"Device: {config.device}")
    print(f"Experiment: {config.experiment_name}")
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize utilities
    logger = Logger(config.log_dir, config.experiment_name)
    visualizer = Visualizer(config.log_dir)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    
    # Log configuration
    logger.log_config(config.to_dict())
    
    # Initialize dataset
    logger.log_info("Loading dataset...")
    dataset = WoodDataset(
        path=config.dataset_path,
        num_class=config.num_class,
        img_size=config.img_size,
        img_scale=config.img_scale,
        brightness=config.aug_brightness,
        scale=config.aug_scale
    )
    
    logger.log_info(f"Dataset classes: {dataset.label_list[:10]}...")  # Show first 10 labels
    
    # Update config with actual number of classes
    config.num_class = dataset.get_num_class()
    logger.log_info(f"Number of classes: {config.num_class}")
    logger.log_info(f"Dataset size: {len(dataset)}")
    
    # Create train/validation split
    train_indices, val_indices = train_test_split(
        list(range(len(dataset.label_list))),
        test_size=0.2,
        stratify=dataset.label_list,
        random_state=42
    )
    
    logger.log_info(f"Train samples: {len(train_indices)}")
    logger.log_info(f"Validation samples: {len(val_indices)}")
    
    # Create data subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    logger.log_info("Initializing model...")
    model = WoodNet(
        num_class=config.num_class,
        img_size=config.img_size,
        layer=config.layer,
        num_hidden=config.num_hidden,
        l2softmax=config.l2softmax,
        dropout_rate=config.dropout_rate
    )
    
    # Log model information
    model_info = model.get_model_info()
    logger.log_model_info(model_info)
    print(model)
    
    # Move model to device
    model.to(config.device)
    
    # Initialize loss function with class weights
    class_weights = dataset.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(config.device)
        logger.log_info(f"Using class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=1000, factor=0.5
    )
    
    # Initialize trainer
    trainer = ClassificationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        scheduler=scheduler,
        logger=logger,
        visualizer=visualizer,
        checkpoint_manager=checkpoint_manager,
        early_stopping_patience=None,  # Disable early stopping for now
        grad_clip_norm=None  # No gradient clipping for now
    )
    
    # Start training
    logger.log_info("Starting training...")
    training_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epoch,
        dataset_for_augmentation=dataset
    )
    
    # Save final configuration
    config_manager.save_config()
    
    logger.log_info("Training completed!")
    logger.close()


if __name__ == "__main__":
    main()
