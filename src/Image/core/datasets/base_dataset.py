"""Base dataset interface for wood classification datasets.

This module provides an abstract base class for implementing dataset loaders
for machine learning tasks, specifically designed for image classification
with PyTorch integration.
"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class BaseDataset(Dataset, ABC):
    """Abstract base class for all image classification datasets.
    
    This class serves as a foundation for implementing custom dataset loaders
    that are compatible with PyTorch's DataLoader. It provides a standardized
    interface for loading and preprocessing image data for machine learning tasks.
    
    The class inherits from both torch.utils.data.Dataset and ABC (Abstract Base Class)
    to ensure that subclasses implement the required methods while maintaining
    compatibility with PyTorch's data loading utilities.
    
    Attributes:
        dataset_path (str): Path to the root directory containing the dataset.
        num_class (int): Number of distinct classes in the dataset.
        img_size (int): Target size for image preprocessing (width and height).
        dataset_params (dict): Additional parameters specific to the dataset implementation.
        fpath_list (list): List of file paths to individual images in the dataset.
        label_list (list): List of corresponding labels for each image.
        weight (Optional[torch.Tensor]): Class weights for handling imbalanced datasets.
    
    Example:
        ```python
        class CustomDataset(BaseDataset):
            def __init__(self, dataset_path: str, **kwargs):
                super().__init__(dataset_path, num_class=10, img_size=224, **kwargs)
                # Custom initialization logic here
                
            def __len__(self) -> int:
                return len(self.fpath_list)
                
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                # Custom data loading logic here
                pass
        ```
    """

    def __init__(
        self, dataset_path: str, num_class: int = 0, img_size: int = 256, **kwargs
    ):
        """Initialize the base dataset with core parameters.

        Sets up the fundamental attributes required by all dataset implementations,
        including dataset path, class count, and image dimensions. Additional
        dataset-specific parameters can be passed through kwargs.

        Args:
            dataset_path (str): Absolute or relative path to the dataset directory.
                               This should point to the root folder containing all
                               dataset files and subdirectories.
            num_class (int, optional): Total number of classes in the dataset.
                                     Defaults to 0, should be overridden by subclasses.
            img_size (int, optional): Target dimension for square image resizing.
                                    Images will be resized to (img_size, img_size).
                                    Defaults to 256 pixels.
            **kwargs: Additional keyword arguments specific to dataset implementations.
                     These are stored in dataset_params for use by subclasses.

        Note:
            Subclasses are responsible for populating fpath_list and label_list
            with actual file paths and corresponding labels during initialization.
        """
        super(BaseDataset, self).__init__()
        self.dataset_path = dataset_path
        self.num_class = num_class
        self.img_size = img_size
        self.dataset_params = kwargs

        # These should be set by subclasses
        self.fpath_list = []
        self.label_list = []
        self.weight = None

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        This method must be implemented by all subclasses to provide the
        dataset size information required by PyTorch's DataLoader for
        batch processing and iteration.
        
        Returns:
            int: Total number of samples/images in the dataset.
            
        Note:
            This method is called by PyTorch's DataLoader to determine
            the number of batches and for validation purposes.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single data sample by index.
        
        This method must be implemented by all subclasses to define how
        individual samples are loaded, preprocessed, and returned. It's
        the core method called by PyTorch's DataLoader during training
        and inference.

        Args:
            idx (int): Index of the sample to retrieve. Must be in the range
                      [0, len(dataset)-1]. The DataLoader ensures this constraint.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - image (torch.Tensor): Preprocessed image tensor with shape
                                       (C, H, W) where C is channels, H is height,
                                       and W is width.
                - label (torch.Tensor): Corresponding label tensor, typically
                                       a scalar tensor for classification tasks.
                                       
        Raises:
            IndexError: If idx is out of bounds for the dataset.
            
        Note:
            Implementations should handle all necessary preprocessing including:
            - Image loading from file
            - Resizing and normalization
            - Data augmentation (if applicable)
            - Tensor conversion
        """
        pass

    def get_num_class(self) -> int:
        """Get the number of classes in the dataset.
        
        This method provides access to the class count information, which
        is essential for configuring neural network output layers and
        evaluation metrics.

        Returns:
            int: Total number of distinct classes in the dataset.
                Returns 0 if not properly initialized by subclass.
                
        Example:
            >>> dataset = SomeDataset("/path/to/data", num_class=10)
            >>> print(dataset.get_num_class())
            10
        """
        return self.num_class

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights for handling imbalanced datasets.
        
        Returns pre-computed class weights that can be used with loss functions
        to address class imbalance issues. These weights are typically computed
        based on the inverse frequency of each class in the training set.

        Returns:
            Optional[torch.Tensor]: Tensor containing weight for each class,
                                   or None if weights are not computed.
                                   Shape: (num_classes,) if present.
                                   
        Example:
            >>> weights = dataset.get_class_weights()
            >>> if weights is not None:
            ...     criterion = nn.CrossEntropyLoss(weight=weights)
            
        Note:
            Class weights are computed during dataset initialization by
            subclasses when handling imbalanced data is required.
        """
        return self.weight

    def get_dataset_info(self) -> dict:
        """Get comprehensive information about the dataset configuration.
        
        Returns a dictionary containing all relevant metadata about the dataset,
        including paths, dimensions, class information, and custom parameters.
        This is useful for logging, debugging, and experiment tracking.

        Returns:
            dict: Dictionary containing the following keys:
                - dataset_name (str): Name of the dataset class
                - path (str): Path to the dataset directory
                - num_class (int): Number of classes
                - img_size (int): Target image size
                - total_samples (int): Total number of samples
                - dataset_params (dict): Additional parameters
                
        Example:
            >>> info = dataset.get_dataset_info()
            >>> print(f"Dataset: {info['dataset_name']}")
            >>> print(f"Classes: {info['num_class']}")
            >>> print(f"Samples: {info['total_samples']}")
            
        Note:
            The total_samples field calls len(self), so subclasses must
            implement __len__ before this method can be used.
        """
        return {
            "dataset_name": self.__class__.__name__,
            "path": self.dataset_path,
            "num_class": self.num_class,
            "img_size": self.img_size,
            "total_samples": len(self),
            "dataset_params": self.dataset_params,
        }
