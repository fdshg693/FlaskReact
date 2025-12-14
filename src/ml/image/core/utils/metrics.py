"""Metrics calculation utilities."""

from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import classification_report


class MetricsCalculator:
    """Class for calculating various metrics during training and evaluation."""

    def __init__(self, num_classes: Any):
        """Initialize metrics calculator.

        Args:
            num_classes: Number of classification classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.total_loss: float = 0.0
        self.total_correct: int = 0
        self.total_samples: int = 0
        self.all_predictions: list[int] = []
        self.all_targets: list[int] = []
        self.confusion_mat: np.ndarray = np.zeros((self.num_classes, self.num_classes))

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor, loss: float
    ) -> None:
        """Update metrics with batch results.

        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size, num_classes) or (batch_size,)
            loss: Batch loss value
        """
        batch_size = predictions.size(0)

        # Convert to numpy for easier handling
        pred_numpy = predictions.detach().cpu().numpy()
        target_numpy = targets.detach().cpu().numpy()

        # Handle one-hot encoded targets
        if len(target_numpy.shape) > 1 and target_numpy.shape[1] > 1:
            target_indices = np.argmax(target_numpy, axis=1)
        else:
            target_indices = target_numpy.astype(int)

        pred_indices = np.argmax(pred_numpy, axis=1)

        # Update counters
        self.total_loss += loss * batch_size
        self.total_correct += np.sum(pred_indices == target_indices)
        self.total_samples += batch_size

        # Store for detailed analysis
        self.all_predictions.extend(pred_indices.tolist())
        self.all_targets.extend(target_indices.tolist())

        # Update confusion matrix
        self._update_confusion_matrix(pred_indices, target_indices)

    def _update_confusion_matrix(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> None:
        """Update confusion matrix.

        Args:
            predictions: Predicted class indices
            targets: Target class indices
        """
        for pred, target in zip(predictions, targets):
            if 0 <= target < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_mat[target, pred] += 1

    def get_accuracy(self) -> float:
        """Get overall accuracy.

        Returns:
            Accuracy as float
        """
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    def get_average_loss(self) -> float:
        """Get average loss.

        Returns:
            Average loss as float
        """
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """Get confusion matrix.

        Args:
            normalize: Whether to normalize the matrix

        Returns:
            Confusion matrix
        """
        if normalize:
            # Normalize by true class (row-wise)
            row_sums = self.confusion_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return self.confusion_mat / row_sums
        return self.confusion_mat.copy()

    def get_per_class_accuracy(self) -> np.ndarray:
        """Get per-class accuracy (recall).

        Returns:
            Per-class accuracy array
        """
        diag = np.diag(self.confusion_mat)
        class_totals = self.confusion_mat.sum(axis=1)
        # Avoid division by zero
        class_totals[class_totals == 0] = 1
        return diag / class_totals

    def get_per_class_precision(self) -> np.ndarray:
        """Get per-class precision.

        Returns:
            Per-class precision array
        """
        diag = np.diag(self.confusion_mat)
        pred_totals = self.confusion_mat.sum(axis=0)
        # Avoid division by zero
        pred_totals[pred_totals == 0] = 1
        return diag / pred_totals

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics report.

        Returns:
            Dictionary containing various metrics
        """
        if len(self.all_targets) == 0:
            return {}

        # Calculate metrics using sklearn
        try:
            report = classification_report(
                self.all_targets,
                self.all_predictions,
                output_dict=True,
                zero_division=0,
            )
        except Exception as e:
            print(f"Warning: Could not generate classification report: {e}")
            report = {}

        return {
            "accuracy": self.get_accuracy(),
            "average_loss": self.get_average_loss(),
            "confusion_matrix": self.get_confusion_matrix(),
            "normalized_confusion_matrix": self.get_confusion_matrix(normalize=True),
            "per_class_accuracy": self.get_per_class_accuracy(),
            "per_class_precision": self.get_per_class_precision(),
            "classification_report": report,
            "total_samples": self.total_samples,
        }

    @staticmethod
    def calculate_top_k_accuracy(
        predictions: torch.Tensor, targets: torch.Tensor, k: int = 5
    ) -> float:
        """Calculate top-k accuracy.

        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels
            k: Top-k parameter

        Returns:
            Top-k accuracy
        """
        with torch.no_grad():
            batch_size = targets.size(0)

            # Handle one-hot encoded targets
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets = torch.argmax(targets, dim=1)

            # Get top-k predictions
            _, pred = predictions.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            return (correct_k / batch_size).item()
