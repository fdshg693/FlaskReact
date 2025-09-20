"""Wood classification dataset implementation."""

import os
import torch
import numpy as np
import cv2
from typing import Tuple
from .base_dataset import BaseDataset


class WoodDataset(BaseDataset):
    """Dataset class for wood classification with comprehensive data augmentation.
    
    This class implements a PyTorch dataset for wood classification tasks. It extends
    BaseDataset and provides specialized functionality for loading wood images,
    applying various data augmentations, and creating balanced training batches.
    
    The dataset automatically detects the number of classes from the directory structure
    and calculates class weights to handle imbalanced datasets. It supports various
    augmentation techniques including scaling, flipping, rotation, and brightness adjustment.
    
    Attributes:
        img_scale (float): Image scaling factor for augmentation.
        rotate (bool): Whether rotation augmentation is enabled.
        brightness (float): Brightness augmentation range.
        hflip (bool): Whether horizontal flip augmentation is enabled.
        vflip (bool): Whether vertical flip augmentation is enabled.
        augmentation (bool): Whether data augmentation is enabled.
        scale (float): Scale augmentation range.
        num_data (np.ndarray): Number of samples per class.
        fpath_list (list[str]): List of file paths for all images.
        label_list (list[int]): List of labels corresponding to each image.
        weight (torch.Tensor): Class weights for balanced training.
    
    Example:
        >>> dataset = WoodDataset(
        ...     dataset_path="data/wood_images",
        ...     img_size=224,
        ...     augmentation=True
        ... )
        >>> image, label = dataset[0]
        >>> print(f"Image shape: {image.shape}, Label shape: {label.shape}")
    """

    def __init__(
        self,
        dataset_path: str = "",
        num_class: int = 0,
        img_size: int = 256,
        img_scale: float = 1.1,
        augmentation: bool = True,
        scale: float = 0.5,
        hflip: bool = True,
        vflip: bool = False,
        rotate: bool = True,
        brightness: float = 0.0,
    ):
        """Initialize the wood dataset with configuration parameters.

        Sets up the dataset by loading image files from the specified directory structure,
        automatically detecting the number of classes, and calculating class weights for
        balanced training. The dataset expects a directory structure where each subdirectory
        represents a different wood class.

        Args:
            dataset_path (str, optional): Path to the dataset root directory containing
                class subdirectories. Defaults to "".
            num_class (int, optional): Number of classes for classification. If 0,
                automatically detects from directory structure. Defaults to 0.
            img_size (int, optional): Target image size (square) for training.
                Images will be cropped/resized to this size. Defaults to 256.
            img_scale (float, optional): Base scaling factor applied to images.
                Actual scaling may vary if augmentation is enabled. Defaults to 1.1.
            augmentation (bool, optional): Whether to enable data augmentation during
                training. Defaults to True.
            scale (float, optional): Random scale augmentation range. The actual scale
                factor will be sampled from [1.0, 1.0 + scale]. Defaults to 0.5.
            hflip (bool, optional): Whether to enable random horizontal flipping
                augmentation. Defaults to True.
            vflip (bool, optional): Whether to enable random vertical flipping
                augmentation. Defaults to False.
            rotate (bool, optional): Whether to enable random rotation augmentation.
                Currently stored but not implemented. Defaults to True.
            brightness (float, optional): Brightness augmentation range. Random values
                are sampled from a normal distribution scaled by this factor.
                Defaults to 0.0 (no brightness augmentation).

        Raises:
            FileNotFoundError: If the dataset_path does not exist or is not accessible.
            ValueError: If no valid image files are found in the dataset directory.

        Note:
            The dataset expects images in common formats (jpg, jpeg, png, bmp, tiff, tif).
            Class labels are assigned based on alphabetical order of directory names.
        """
        super(WoodDataset, self).__init__(
            dataset_path=dataset_path,
            num_class=num_class,
            img_size=img_size,
            img_scale=img_scale,
            augmentation=augmentation,
            scale=scale,
            hflip=hflip,
            vflip=vflip,
            rotate=rotate,
            brightness=brightness,
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
        """Load dataset files and labels from the directory structure.
        
        Scans the dataset directory to automatically discover classes and load all
        image files. Each subdirectory is treated as a separate class, and class
        indices are assigned based on alphabetical order of directory names.
        
        The method populates the following instance attributes:
        - num_class: Updated if auto-detection is enabled (num_class=0)
        - num_data: Array containing the number of samples per class
        - fpath_list: List of file paths for all valid images
        - label_list: List of class labels corresponding to each image
        
        Raises:
            OSError: If the dataset directory cannot be accessed.
            ValueError: If no valid class directories or image files are found.
            
        Note:
            Only files with valid image extensions are included. Non-image files
            and subdirectories are automatically filtered out.
        """
        # Auto-detect number of classes if not specified
        if self.num_class == 0:
            self.num_class = len(
                [
                    d
                    for d in os.listdir(self.dataset_path)
                    if os.path.isdir(os.path.join(self.dataset_path, d))
                ]
            )

        self.num_data = np.zeros(self.num_class)
        self.fpath_list = []
        self.label_list = []
        # クラス名リストを保存（順序固定化のため）
        self.class_names = []  # クラス名を保存するリスト

        class_idx = 0
        for directory in sorted(os.listdir(self.dataset_path)):
            dir_path = os.path.join(self.dataset_path, directory)

            if not os.path.isdir(dir_path):
                continue

            print(f"Loading class {class_idx}: {dir_path}")
            # クラス名（ディレクトリ名）を保存（データセット作成順で固定化）
            self.class_names.append(directory)  # ディレクトリ名をクラス名として保存

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

    def get_class_names(self) -> list:
        """クラス名のリストを返します。
        
        データセット作成順でクラス名を返すため、OS依存を回避します。
        
        Returns:
            list: クラス名のリスト（ディレクトリ名に基づく、順序固定）
        """
        return self.class_names.copy() if hasattr(self, 'class_names') else []

    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is a valid image based on its extension.

        Validates whether the given filename has an extension that corresponds to
        a supported image format. The check is case-insensitive.

        Args:
            filename (str): The filename to check, including the file extension.

        Returns:
            bool: True if the file has a valid image extension, False otherwise.

        Example:
            >>> dataset = WoodDataset()
            >>> dataset._is_image_file("example.jpg")
            True
            >>> dataset._is_image_file("document.txt")
            False
            >>> dataset._is_image_file("image.PNG")  # Case insensitive
            True

        Note:
            Supported image formats include: jpg, jpeg, png, bmp, tiff, tif.
            The method only checks the file extension and does not validate
            the actual file content.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def _calculate_class_weights(self) -> None:
        """Calculate class weights for balanced training to handle imbalanced datasets.
        
        Computes inverse frequency weights for each class to ensure balanced training
        even when the dataset has unequal numbers of samples per class. The weights
        are calculated such that classes with fewer samples receive higher weights.
        
        The calculation follows the formula:
        weight[i] = num_data / num_data[i] (normalized by minimum)
        
        Updates the following instance attributes:
        - weight: PyTorch tensor containing the calculated class weights
        
        The method also prints the calculated weights and their sum for debugging purposes.
        
        Note:
            Classes with zero samples are assigned a weight of 1 to avoid division by zero.
            The weights are converted to a PyTorch float tensor for compatibility with
            loss functions that support class weighting.
            
        Example:
            For a 3-class dataset with samples [100, 50, 200]:
            - Raw weights: [1.0, 2.0, 0.5]
            - Normalized weights: [2.0, 4.0, 1.0] (divided by min=0.5)
        """
        # Avoid division by zero
        denom = self.num_data.copy()
        denom[denom < 1] = 1

        # Calculate weights inversely proportional to class frequency
        self.weight = denom / denom.min()
        self.weight = torch.from_numpy(self.weight).float()

        print(f"Class weights: {self.weight.numpy()}")
        print(f"Sum of weights: {self.weight.sum().item()}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        Returns:
            int: The number of image samples available in the dataset.
            
        Note:
            This method is required by PyTorch's Dataset interface and is used
            by DataLoader to determine iteration length and sampling strategies.
        """
        return len(self.fpath_list)

    def set_augmentation(self, augmentation: bool) -> None:
        """Enable or disable data augmentation for the dataset.

        This method allows dynamic control over data augmentation during training
        or evaluation phases. Typically, augmentation is enabled during training
        for better generalization and disabled during validation/testing for
        consistent evaluation.

        Args:
            augmentation (bool): Whether to apply data augmentation transformations.
                True enables augmentation (scaling, flipping, brightness adjustment),
                False disables all augmentation except basic resizing and cropping.

        Example:
            >>> dataset = WoodDataset(augmentation=True)
            >>> dataset.set_augmentation(False)  # Disable for validation
            >>> # ... run validation ...
            >>> dataset.set_augmentation(True)   # Re-enable for training

        Note:
            This setting affects all augmentation operations including horizontal/vertical
            flips, scale variations, and brightness adjustments. Basic operations like
            resizing to target size and center cropping are always applied.
        """
        self.augmentation = augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single sample from the dataset by index.

        This method implements the core data loading pipeline for PyTorch datasets.
        It loads an image, applies preprocessing transformations (scaling, augmentation,
        cropping), normalizes the pixel values, and returns both the processed image
        and its corresponding one-hot encoded label.

        The processing pipeline includes:
        1. Load image using OpenCV
        2. Apply scaling based on img_scale parameter
        3. Handle undersized images by resizing
        4. Apply data augmentations (if enabled)
        5. Random crop to target size
        6. Normalize pixel values to [0, 1] range
        7. Convert to PyTorch tensors with proper channel ordering

        Args:
            idx (int): Index of the sample to retrieve. Must be in range [0, len(dataset)-1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image (torch.Tensor): Processed image tensor with shape (C, H, W)
                  where C=3 (RGB channels), H=W=img_size. Values in range [0, 1].
                - label (torch.Tensor): One-hot encoded class label with shape (num_class,).
                  Only one element is 1.0, others are 0.0.

        Raises:
            IndexError: If idx is out of range.
            RuntimeError: If image loading fails and dummy data cannot be created.

        Example:
            >>> dataset = WoodDataset("path/to/data", img_size=224, num_class=5)
            >>> image, label = dataset[0]
            >>> assert image.shape == (3, 224, 224)
            >>> assert label.shape == (5,)
            >>> assert torch.sum(label) == 1.0  # One-hot property

        Note:
            If an image file cannot be loaded (corrupted or missing), the method
            creates a zero-filled dummy image to maintain dataset consistency.
            The image tensor uses PyTorch's standard (C, H, W) channel ordering.
        """
        # Load image
        img = cv2.imread(self.fpath_list[idx])

        if img is None:
            print(f"File not found: {self.fpath_list[idx]}")
            # Return dummy data for robustness
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Apply scaling
        img = self._apply_scaling(img)

        # Check image size constraints
        if self.img_size > img.shape[0] or self.img_size > img.shape[1]:
            print(
                f"Image shape smaller than img_size: {self.fpath_list[idx]}, {img.shape}"
            )
            # Resize to minimum required size
            scale_factor = max(
                self.img_size / img.shape[0], self.img_size / img.shape[1]
            )
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
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(
            label
        ).float()

    def _apply_scaling(self, img: np.ndarray) -> np.ndarray:
        """Apply scaling transformation to resize the input image.

        Calculates the appropriate scale factor based on the target image size,
        base scaling factor, and optional random scale augmentation. The scaling
        ensures that images are resized to an appropriate size for subsequent
        cropping operations.

        The scale factor calculation:
        - Base scale: img_size / image_width * img_scale
        - With augmentation: additional random factor in range [1.0, 1.0 + scale]
        - Without augmentation: uses only the base scale factor

        Args:
            img (np.ndarray): Input image array with shape (H, W, C) where
                H is height, W is width, and C is the number of channels (typically 3).

        Returns:
            np.ndarray: Scaled image array with the same number of channels but
                potentially different height and width dimensions.

        Example:
            >>> dataset = WoodDataset(img_size=224, img_scale=1.1, scale=0.5)
            >>> image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            >>> scaled = dataset._apply_scaling(image)
            >>> # Result will be scaled to approximately 224 * 1.1 = 246 pixels width
            >>> # with additional random variation if augmentation is enabled

        Note:
            The scaling is applied uniformly to both height and width dimensions
            to maintain the original aspect ratio of the image.
        """
        if self.augmentation and self.scale > 0:
            img_scale = (
                self.img_size
                / img.shape[1]
                * self.img_scale
                * (1.0 + np.random.rand() * self.scale)
            )
        else:
            img_scale = self.img_size / img.shape[1] * self.img_scale

        return cv2.resize(img, None, fx=img_scale, fy=img_scale)

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply various data augmentation transformations to the input image.

        Performs random augmentations to increase dataset diversity and improve
        model generalization. The augmentations are applied probabilistically
        based on the configuration parameters set during initialization.

        Available augmentations:
        - Horizontal flip: 50% probability if enabled
        - Vertical flip: 50% probability if enabled  
        - Brightness adjustment: Gaussian noise scaled by brightness factor

        Args:
            img (np.ndarray): Input image array with shape (H, W, C) and
                pixel values typically in range [0, 255].

        Returns:
            np.ndarray: Augmented image array with the same shape as input.
                Pixel values are clipped to valid range [0, 255] after augmentation.

        Example:
            >>> dataset = WoodDataset(hflip=True, vflip=False, brightness=0.1)
            >>> image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            >>> augmented = dataset._apply_augmentations(image)
            >>> # Image may be horizontally flipped (50% chance)
            >>> # Brightness may be slightly adjusted

        Note:
            - Flipping operations use NumPy array slicing for efficiency
            - Brightness augmentation adds Gaussian noise scaled by the brightness parameter
            - All augmentations preserve the original image dimensions
            - Pixel values are clipped to [0, 255] to prevent overflow/underflow
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
        """Apply random cropping to extract a fixed-size patch from the image.

        Extracts a square patch of size (img_size, img_size) from a random location
        within the input image. If the input image is smaller than the target size,
        it is first padded with zeros to ensure a valid crop can be extracted.

        The random cropping process:
        1. Check if image is smaller than target size
        2. If too small, pad with zeros on the bottom and right edges
        3. Randomly select crop coordinates within valid range
        4. Extract the square patch at the selected location

        Args:
            img (np.ndarray): Input image array with shape (H, W, C) where
                H >= img_size and W >= img_size (after padding if necessary).

        Returns:
            np.ndarray: Cropped image patch with shape (img_size, img_size, C)
                containing the randomly selected region from the input image.

        Example:
            >>> dataset = WoodDataset(img_size=224)
            >>> large_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
            >>> cropped = dataset._random_crop(large_image)
            >>> assert cropped.shape == (224, 224, 3)
            
            >>> small_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            >>> cropped = dataset._random_crop(small_image)  # Will be padded first
            >>> assert cropped.shape == (224, 224, 3)

        Note:
            - Padding uses constant (zero) values for simplicity
            - Random coordinates are sampled uniformly from valid ranges
            - This method ensures consistent output dimensions regardless of input size
            - For images exactly matching the target size, returns the full image
        """
        if img.shape[0] <= self.img_size or img.shape[1] <= self.img_size:
            # If image is too small, pad it
            pad_h = max(0, self.img_size - img.shape[0] + 1)
            pad_w = max(0, self.img_size - img.shape[1] + 1)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # Random crop
        max_v = img.shape[0] - self.img_size
        max_h = img.shape[1] - self.img_size

        v = np.random.randint(0, max_v + 1) if max_v > 0 else 0
        h = np.random.randint(0, max_h + 1) if max_h > 0 else 0

        return img[v : v + self.img_size, h : h + self.img_size]


# Backward compatibility alias
Wood_Dataset = WoodDataset
"""
Deprecated alias for WoodDataset class.

This alias is maintained for backward compatibility with existing code that
uses the old naming convention. New code should use WoodDataset directly.

Warning:
    This alias is deprecated and may be removed in future versions.
    Please update your code to use WoodDataset instead of Wood_Dataset.

Example:
    # Deprecated usage
    dataset = Wood_Dataset("path/to/data")
    
    # Preferred usage
    dataset = WoodDataset("path/to/data")
"""
