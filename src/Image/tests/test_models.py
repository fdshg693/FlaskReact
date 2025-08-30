"""Test models module."""
import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.wood_net import WoodNet


class TestWoodNet(unittest.TestCase):
    """Test cases for WoodNet model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_class = 5
        self.img_size = 128
        self.batch_size = 4
        self.model = WoodNet(
            num_class=self.num_class,
            img_size=self.img_size,
            layer=3,
            num_hidden=1024,
            dropout_rate=0.2
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.num_class, self.num_class)
        self.assertEqual(self.model.img_size, self.img_size)
        self.assertIsInstance(self.model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        # Create random input
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_class)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_info(self):
        """Test model info method."""
        info = self.model.get_model_info()
        
        self.assertIn('model_name', info)
        self.assertIn('num_class', info)
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        
        self.assertEqual(info['num_class'], self.num_class)
        self.assertEqual(info['model_name'], 'WoodNet')
    
    def test_save_load_model(self):
        """Test save and load functionality."""
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            self.model.save_model(temp_path)
            
            # Create new model and load
            new_model = WoodNet(
                num_class=self.num_class,
                img_size=self.img_size
            )
            new_model.load_model(temp_path)
            
            # Test that models produce same output
            x = torch.randn(1, 3, self.img_size, self.img_size)
            with torch.no_grad():
                output1 = self.model(x)
                output2 = new_model(x)
            
            self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
