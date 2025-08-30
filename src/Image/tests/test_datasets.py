"""Test cases for datasets module."""
import os
import tempfile
import shutil
import pytest
import torch
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.base_dataset import BaseDataset
from src.datasets.wood_dataset import WoodDataset


class MockDataset(BaseDataset):
    """Mock implementation of BaseDataset for testing."""
    
    def __init__(self, path: str, num_class: int = 3, img_size: int = 256, **kwargs):
        super(MockDataset, self).__init__(path, num_class, img_size, **kwargs)
        self.fpath_list = [f"image_{i}.jpg" for i in range(10)]
        self.label_list = [i % num_class for i in range(10)]
    
    def __len__(self) -> int:
        return len(self.fpath_list)
    
    def __getitem__(self, idx: int):
        # Return dummy tensors
        image = torch.randn(3, self.img_size, self.img_size)
        label = torch.zeros(self.num_class)
        label[self.label_list[idx]] = 1.0
        return image, label


class TestBaseDataset:
    """Test cases for BaseDataset class."""
    
    def test_init(self):
        """Test BaseDataset initialization."""
        dataset = MockDataset(
            path="/test/path",
            num_class=5,
            img_size=224,
            custom_param="test"
        )
        
        assert dataset.path == "/test/path"
        assert dataset.num_class == 5
        assert dataset.img_size == 224
        assert dataset.dataset_params["custom_param"] == "test"
        assert len(dataset.fpath_list) == 10
        assert len(dataset.label_list) == 10
    
    def test_get_num_class(self):
        """Test get_num_class method."""
        dataset = MockDataset("/test/path", num_class=7)
        assert dataset.get_num_class() == 7
    
    def test_get_class_weights(self):
        """Test get_class_weights method."""
        dataset = MockDataset("/test/path")
        # Initially None
        assert dataset.get_class_weights() is None
        
        # Set weights and test
        weights = torch.tensor([1.0, 2.0, 3.0])
        dataset.weight = weights
        assert torch.equal(dataset.get_class_weights(), weights)
    
    def test_get_dataset_info(self):
        """Test get_dataset_info method."""
        dataset = MockDataset(
            "/test/path",
            num_class=3,
            img_size=256,
            custom_param="test"
        )
        
        info = dataset.get_dataset_info()
        
        assert info["dataset_name"] == "MockDataset"
        assert info["path"] == "/test/path"
        assert info["num_class"] == 3
        assert info["img_size"] == 256
        assert info["total_samples"] == 10
        assert info["dataset_params"]["custom_param"] == "test"
    
    def test_len(self):
        """Test __len__ method."""
        dataset = MockDataset("/test/path")
        assert len(dataset) == 10
    
    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = MockDataset("/test/path", num_class=3, img_size=128)
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.shape == (3, 128, 128)
        assert label.shape == (3,)
        assert label.sum().item() == 1.0  # One-hot encoded


class TestWoodDataset:
    """Test cases for WoodDataset class."""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory structure."""
        temp_dir = tempfile.mkdtemp()
        
        # Create class directories
        class_dirs = ["class_0", "class_1", "class_2"]
        for class_dir in class_dirs:
            class_path = os.path.join(temp_dir, class_dir)
            os.makedirs(class_path)
            
            # Create dummy images for each class
            for i in range(3):
                img_path = os.path.join(class_path, f"image_{i}.jpg")
                # Create a simple test image
                test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_img)
        
        # Create non-image files (should be ignored)
        with open(os.path.join(temp_dir, "class_0", "readme.txt"), "w") as f:
            f.write("This is not an image")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_init_auto_detect_classes(self, temp_dataset_dir):
        """Test WoodDataset initialization with auto class detection."""
        dataset = WoodDataset(
            path=temp_dataset_dir,
            num_class=0,  # Auto-detect
            img_size=64,
            augmentation=False
        )
        
        assert dataset.num_class == 3
        assert len(dataset.fpath_list) == 9  # 3 classes × 3 images
        assert len(dataset.label_list) == 9
        assert dataset.img_size == 64
        assert not dataset.augmentation
    
    def test_init_specified_classes(self, temp_dataset_dir):
        """Test WoodDataset initialization with specified number of classes."""
        dataset = WoodDataset(
            path=temp_dataset_dir,
            num_class=2,  # Only load first 2 classes
            img_size=128,
            augmentation=True
        )
        
        assert dataset.num_class == 2
        assert len(dataset.fpath_list) == 6  # 2 classes × 3 images
        assert len(dataset.label_list) == 6
        assert dataset.augmentation
    
    def test_is_image_file(self, temp_dataset_dir):
        """Test _is_image_file method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=1, augmentation=False)
        
        assert dataset._is_image_file("test.jpg")
        assert dataset._is_image_file("test.jpeg")
        assert dataset._is_image_file("test.png")
        assert dataset._is_image_file("test.bmp")
        assert dataset._is_image_file("test.tiff")
        assert dataset._is_image_file("test.tif")
        assert dataset._is_image_file("Test.JPG")  # Case insensitive
        
        assert not dataset._is_image_file("test.txt")
        assert not dataset._is_image_file("test.pdf")
        assert not dataset._is_image_file("test")
    
    def test_calculate_class_weights(self, temp_dataset_dir):
        """Test _calculate_class_weights method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=3, augmentation=False)
        
        assert dataset.weight is not None
        assert isinstance(dataset.weight, torch.Tensor)
        assert dataset.weight.shape == (3,)
        assert torch.all(dataset.weight > 0)
    
    def test_len(self, temp_dataset_dir):
        """Test __len__ method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=3, augmentation=False)
        assert len(dataset) == 9
    
    def test_set_augmentation(self, temp_dataset_dir):
        """Test set_augmentation method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=1, augmentation=True)
        
        assert dataset.augmentation is True
        
        dataset.set_augmentation(False)
        assert dataset.augmentation is False
        
        dataset.set_augmentation(True)
        assert dataset.augmentation is True
    
    def test_getitem_without_augmentation(self, temp_dataset_dir):
        """Test __getitem__ method without augmentation."""
        dataset = WoodDataset(
            temp_dataset_dir,
            num_class=3,
            img_size=64,
            augmentation=False
        )
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.shape == (3, 64, 64)  # (C, H, W)
        assert label.shape == (3,)  # One-hot encoded
        assert label.sum().item() == 1.0
        assert torch.all(image >= 0) and torch.all(image <= 1)  # Normalized
    
    def test_getitem_with_augmentation(self, temp_dataset_dir):
        """Test __getitem__ method with augmentation."""
        dataset = WoodDataset(
            temp_dataset_dir,
            num_class=3,
            img_size=64,
            augmentation=True,
            hflip=True,
            vflip=True,
            brightness=0.1
        )
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert label.shape == (3,)
        assert label.sum().item() == 1.0
        assert torch.all(image >= 0) and torch.all(image <= 1)
    
    @patch('cv2.imread')
    def test_getitem_file_not_found(self, mock_imread, temp_dataset_dir):
        """Test __getitem__ method when file is not found."""
        mock_imread.return_value = None
        
        dataset = WoodDataset(temp_dataset_dir, num_class=1, augmentation=False)
        image, label = dataset[0]
        
        # Should return dummy data
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert image.shape == (3, 256, 256)  # Default img_size
    
    def test_apply_scaling(self, temp_dataset_dir):
        """Test _apply_scaling method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=1, augmentation=False)
        
        # Create test image
        test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        scaled_img = dataset._apply_scaling(test_img)
        
        assert isinstance(scaled_img, np.ndarray)
        assert len(scaled_img.shape) == 3
        assert scaled_img.shape[2] == 3
    
    def test_apply_augmentations(self, temp_dataset_dir):
        """Test _apply_augmentations method."""
        dataset = WoodDataset(
            temp_dataset_dir,
            num_class=1,
            augmentation=True,
            hflip=True,
            vflip=True,
            brightness=0.1
        )
        
        # Create test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        augmented_img = dataset._apply_augmentations(test_img)
        
        assert isinstance(augmented_img, np.ndarray)
        assert augmented_img.shape == test_img.shape
    
    def test_random_crop(self, temp_dataset_dir):
        """Test _random_crop method."""
        dataset = WoodDataset(temp_dataset_dir, num_class=1, img_size=64, augmentation=False)
        
        # Test with larger image
        large_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cropped_img = dataset._random_crop(large_img)
        
        assert cropped_img.shape == (64, 64, 3)
        
        # Test with smaller image (should be padded)
        small_img = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        padded_img = dataset._random_crop(small_img)
        
        assert padded_img.shape == (64, 64, 3)
    
    def test_get_dataset_info(self, temp_dataset_dir):
        """Test get_dataset_info method."""
        dataset = WoodDataset(
            temp_dataset_dir,
            num_class=3,
            img_size=128,
            augmentation=True
        )
        
        info = dataset.get_dataset_info()
        
        assert info["dataset_name"] == "WoodDataset"
        assert info["path"] == temp_dataset_dir
        assert info["num_class"] == 3
        assert info["img_size"] == 128
        assert info["total_samples"] == 9
        assert "augmentation" in info["dataset_params"]
    
    def test_backward_compatibility_alias(self, temp_dataset_dir):
        """Test backward compatibility alias."""
        from src.datasets.wood_dataset import Wood_Dataset
        
        dataset = Wood_Dataset(temp_dataset_dir, num_class=1, augmentation=False)
        assert isinstance(dataset, WoodDataset)


class TestDatasetIntegration:
    """Integration tests for datasets."""
    
    @pytest.fixture
    def sample_dataset_dir(self):
        """Create a more realistic sample dataset."""
        temp_dir = tempfile.mkdtemp()
        
        # Create realistic class structure
        classes = ["pine", "oak", "birch"]
        for class_name in classes:
            class_path = os.path.join(temp_dir, class_name)
            os.makedirs(class_path)
            
            # Create images with different sizes
            for i in range(5):
                img_path = os.path.join(class_path, f"{class_name}_{i:03d}.jpg")
                # Create images with varying sizes
                height = np.random.randint(150, 400)
                width = np.random.randint(150, 400)
                test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_img)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_dataset_loading_and_iteration(self, sample_dataset_dir):
        """Test complete dataset loading and iteration."""
        dataset = WoodDataset(
            sample_dataset_dir,
            num_class=0,  # Auto-detect
            img_size=128,
            augmentation=True
        )
        
        # Test basic properties
        assert dataset.num_class == 3
        assert len(dataset) == 15  # 3 classes × 5 images
        
        # Test iteration through dataset
        all_labels = []
        for i in range(len(dataset)):
            image, label = dataset[i]
            
            # Check tensor properties
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert image.shape == (3, 128, 128)
            assert label.shape == (3,)
            assert label.sum().item() == 1.0
            
            # Track labels for distribution check
            all_labels.append(label.argmax().item())
        
        # Check label distribution
        unique_labels = set(all_labels)
        assert len(unique_labels) == 3
        assert unique_labels == {0, 1, 2}
    
    def test_dataset_with_dataloader(self, sample_dataset_dir):
        """Test dataset compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = WoodDataset(
            sample_dataset_dir,
            num_class=3,
            img_size=64,
            augmentation=False
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test one batch
        for batch_images, batch_labels in dataloader:
            assert batch_images.shape == (4, 3, 64, 64)
            assert batch_labels.shape == (4, 3)
            assert torch.all(batch_labels.sum(dim=1) == 1.0)  # All one-hot
            break
    
    def test_dataset_class_weights_calculation(self, sample_dataset_dir):
        """Test class weights calculation with realistic data."""
        dataset = WoodDataset(
            sample_dataset_dir,
            num_class=3,
            augmentation=False
        )
        
        weights = dataset.get_class_weights()
        assert weights is not None
        assert len(weights) == 3
        assert torch.all(weights > 0)
        
        # Since all classes have equal samples, weights should be equal
        assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
