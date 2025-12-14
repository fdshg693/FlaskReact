from __future__ import annotations

from typing import Any

import torch

from ml.image.core.evaluation.model_evaluator import predict_image_data


def predict_image_service(checkpoint_path: str, img_tensor: torch.Tensor) -> Any:
    return predict_image_data(checkpoint_path=checkpoint_path, img_tensor=img_tensor)
