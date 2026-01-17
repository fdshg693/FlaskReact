import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNetwork(nn.Module):
    """小さな2層の全結合ニューラルネットワーク。

    This is a shared implementation extracted from other modules to avoid
    duplication.
    """

    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3
    ) -> None:
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fully_connected_layer_1(x))
        return self.fully_connected_layer_2(x)
