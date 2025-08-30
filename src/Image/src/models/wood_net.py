"""Wood classification neural network model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class WoodNet(BaseModel):
    """Neural network for wood classification."""
    
    def __init__(self, num_class: int = 0, img_size: int = 128, layer: int = 4, 
                 num_hidden: int = 4096, l2softmax: bool = True, dropout_rate: float = 0.5):
        """Initialize WoodNet model.
        
        Args:
            num_class: Number of classification classes
            img_size: Input image size
            layer: Number of convolutional layers
            num_hidden: Number of hidden units in FC layer
            l2softmax: Whether to use L2 normalization before softmax
            dropout_rate: Dropout probability
        """
        super(WoodNet, self).__init__(
            num_class=num_class,
            img_size=img_size,
            layer=layer,
            num_hidden=num_hidden,
            l2softmax=l2softmax,
            dropout_rate=dropout_rate
        )
        
        self.alpha = 10
        self.img_size = img_size
        self.layer = layer
        self.l2softmax = l2softmax
        self.debug = False
        self.dropout = nn.Dropout(dropout_rate)
        
        # Build convolutional layers
        self.conv = self._build_conv_layers()
        
        # Calculate output size after convolutions
        out_size = self._calculate_final_conv_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear((10 + layer * 2) ** 2 * out_size ** 2, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_class, bias=False)
        
        # Activation functions
        self.mish = nn.Mish()
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build convolutional layers.
        
        Returns:
            ModuleList of convolutional layers
        """
        conv_layers = []
        
        # First layer: 3 -> (10*10) channels
        conv_layers.append(nn.Conv2d(3, 10 * 10, (3, 3), stride=2, padding=1))
        
        # Additional layers
        for i in range(self.layer):
            in_channels = (10 + i * 2) ** 2
            out_channels = (10 + (i + 1) * 2) ** 2
            conv_layers.append(nn.Conv2d(in_channels, out_channels, (3, 3), stride=2, padding=1))
        
        return nn.ModuleList(conv_layers)
    
    def _calculate_final_conv_size(self) -> int:
        """Calculate output size after all convolutional layers.
        
        Returns:
            Final spatial dimension size
        """
        size = self.img_size
        for _ in range(len(self.conv)):
            size = self._calc_output_size(size, 3, 2, 1)
        return size
    
    def _calc_output_size(self, img_size: int, fsize: int, stride: int, padding: int) -> int:
        """Calculate output size for a single conv layer.
        
        Args:
            img_size: Input size
            fsize: Filter size
            stride: Stride
            padding: Padding
            
        Returns:
            Output size
        """
        return (img_size + 2 * padding - fsize) // stride + 1
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        # Initialize conv layers
        for conv_layer in self.conv:
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            nn.init.zeros_(conv_layer.bias)
        
        # Initialize FC layers
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, img_size, img_size)
            
        Returns:
            Output tensor of shape (batch_size, num_class)
        """
        # Convolutional layers
        h = x
        for conv_layer in self.conv:
            h = conv_layer(h)
            h = self.mish(h)
        
        if self.debug:
            print(f"Conv output shape: {h.shape}")
        
        # Flatten for FC layers
        h = h.view(h.shape[0], -1)
        
        # First FC layer with dropout
        h = self.fc1(h)
        h = self.mish(h)
        h = self.dropout(h)
        
        # Second FC layer
        h = self.fc2(h)
        
        # L2 normalization if enabled
        if self.l2softmax:
            l2_norm = torch.sqrt((h ** 2).sum(dim=1, keepdim=True))
            h = self.alpha * h / l2_norm
        
        return h


# Backward compatibility alias
Wood_Net = WoodNet
