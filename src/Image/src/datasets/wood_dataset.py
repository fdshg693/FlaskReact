"""Wood classification dataset implementation."""
import os
import torch
import numpy as np
import cv2
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class WoodDataset(BaseDataset):
    """Dataset class for wood classification."""
    
    def __init__(self, path: str = '', num_class: int = 0, 
                img_size: int = 256, img_scale: float = 1.1, augmentation: bool = True,
                scale: float = 0.5, hflip: bool = True, vflip: bool = False, 
                rotate: bool = True, brightness: float = 0.0):
        """Initialize wood dataset.
        
        Args:
            path: Dataset path
            num_class: Number of classes (0 for auto-detection)
            img_size: Image size for training
            img_scale: Image scaling factor
            augmentation: Whether to apply data augmentation
            scale: Scale augmentation range
            hflip: Whether to apply horizontal flip
            vflip: Whether to apply vertical flip
            rotate: Whether to apply rotation
            brightness: Brightness augmentation range
        """
        super(WoodDataset, self).__init__(
            path=path,
            num_class=num_class,
            img_size=img_size,
            img_scale=img_scale,
            augmentation=augmentation,
            scale=scale,
            hflip=hflip,
            vflip=vflip,
            rotate=rotate,
            brightness=brightness
        )
        
        # Store augmentation parameters
        self.img_scale = img_scale
        self.rotate = rotate
        self.brightness = brightness
        self.hflip = hflip
        self.vflip = vflip
        self.augmentation = augmentation
        self.scale = scale
        
        # Initialize dataset
        self._load_dataset()
        self._calculate_class_weights()
    
    def _load_dataset(self) -> None:
        """Load dataset files and labels."""
        # Auto-detect number of classes if not specified
        if self.num_class == 0:
            self.num_class = len([d for d in os.listdir(self.path) 
                                if os.path.isdir(os.path.join(self.path, d))])
        
        self.num_data = np.zeros(self.num_class)
        self.fpath_list = []
        self.label_list = []
        
        class_idx = 0
        for directory in sorted(os.listdir(self.path)):
            dir_path = os.path.join(self.path, directory)
            
            if not os.path.isdir(dir_path):
                continue
            
            print(f"Loading class {class_idx}: {dir_path}")
            
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                # Check if file is an image
                if not self._is_image_file(filename):
                    continue
                
                self.fpath_list.append(file_path)
                self.label_list.append(class_idx)
                self.num_data[class_idx] += 1
            
            class_idx += 1
            if class_idx >= self.num_class:
                break
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if file is an image.
        
        Args:
            filename: File name to check
            
        Returns:
            True if file is an image
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def _calculate_class_weights(self) -> None:
        """Calculate class weights for balanced training."""
        # Avoid division by zero
        denom = self.num_data.copy()
        denom[denom < 1] = 1
        
        # Calculate weights inversely proportional to class frequency
        self.weight = denom / denom.min()
        self.weight = torch.from_numpy(self.weight).float()
        
        print(f'Class weights: {self.weight.numpy()}')
        print(f'Sum of weights: {self.weight.sum().item()}')
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.fpath_list)
    
    def set_augmentation(self, augmentation: bool) -> None:
        """Set augmentation mode.
        
        Args:
            augmentation: Whether to apply augmentation
        """
        self.augmentation = augmentation
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image tensor, label tensor)
        """
        # Load image
        img = cv2.imread(self.fpath_list[idx])
        
        if img is None:
            print(f'File not found: {self.fpath_list[idx]}')
            # Return dummy data for robustness
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Apply scaling
        img = self._apply_scaling(img)
        
        # Check image size constraints
        if self.img_size > img.shape[0] or self.img_size > img.shape[1]:
            print(f'Image shape smaller than img_size: {self.fpath_list[idx]}, {img.shape}')
            # Resize to minimum required size
            scale_factor = max(self.img_size / img.shape[0], self.img_size / img.shape[1])
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
        
        # Apply augmentations
        if self.augmentation:
            img = self._apply_augmentations(img)
        
        # Random crop to target size
        img = self._random_crop(img)
        
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # Create one-hot label
        label = np.eye(self.num_class)[int(self.label_list[idx])]
        
        # Convert to tensors and transpose image for PyTorch (C, H, W)
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(label).float()
    
    def _apply_scaling(self, img: np.ndarray) -> np.ndarray:
        """Apply scaling to image.
        
        Args:
            img: Input image
            
        Returns:
            Scaled image
        """
        if self.augmentation and self.scale > 0:
            img_scale = (self.img_size / img.shape[1] * self.img_scale * 
                        (1.0 + np.random.rand() * self.scale))
        else:
            img_scale = self.img_size / img.shape[1] * self.img_scale
        
        return cv2.resize(img, None, fx=img_scale, fy=img_scale)
    
    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentations to image.
        
        Args:
            img: Input image
            
        Returns:
            Augmented image
        """
        # Horizontal flip
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1, :]
        
        # Vertical flip
        if self.vflip and np.random.rand() < 0.5:
            img = img[::-1, :, :]
        
        # Brightness augmentation
        if self.brightness != 0.0:
            brightness_factor = np.random.normal(0, 1.0, 1)
            img = (img + self.brightness * brightness_factor).clip(0, 255)
        
        return img
    
    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        """Apply random crop to image.
        
        Args:
            img: Input image
            
        Returns:
            Cropped image
        """
        if img.shape[0] <= self.img_size or img.shape[1] <= self.img_size:
            # If image is too small, pad it
            pad_h = max(0, self.img_size - img.shape[0] + 1)
            pad_w = max(0, self.img_size - img.shape[1] + 1)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        # Random crop
        max_v = img.shape[0] - self.img_size
        max_h = img.shape[1] - self.img_size
        
        v = np.random.randint(0, max_v + 1) if max_v > 0 else 0
        h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        
        return img[v:v + self.img_size, h:h + self.img_size]


# Backward compatibility alias
Wood_Dataset = WoodDataset
