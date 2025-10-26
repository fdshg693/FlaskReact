from __future__ import annotations

from typing import Any
from image.core.evaluation.evaluator import predict_image_data
import torch


def predict_image_service(checkpoint_path: str, img_tensor: torch.Tensor) -> Any:
    return predict_image_data(checkpoint_path=checkpoint_path, img_tensor=img_tensor)
