"""Base dataset interface for wood classification datasets."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets."""

    dataset_path: str
    num_class: int
    img_size: int
    dataset_params: dict[str, object]
    fpath_list: list[str]
    label_list: list[int]
    weight: Optional[torch.Tensor]

    def __init__(
        self,
        dataset_path: str,
        num_class: int = 0,
        img_size: int = 256,
        **kwargs: object,
    ):
        """Initialize base dataset.

        Args:
            dataset_path: Dataset path
            num_class: Number of classes
            img_size: Image size
            **kwargs: Additional dataset-specific parameters
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
        """Return dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item by index.

        Args:
            idx: Item index

        Returns:
            Tuple of (image, label)
        """
        pass

    def get_num_class(self) -> int:
        """Get number of classes.

        Returns:
            Number of classes
        """
        return self.num_class

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights for balanced training.

        Returns:
            Class weights tensor or None
        """
        return self.weight

    def get_dataset_info(self) -> dict:
        """Get dataset information.

        Returns:
            Dictionary containing dataset info
        """
        return {
            "dataset_name": self.__class__.__name__,
            "path": self.dataset_path,
            "num_class": self.num_class,
            "img_size": self.img_size,
            "total_samples": len(self),
            "dataset_params": self.dataset_params,  # type: ignore
        }
