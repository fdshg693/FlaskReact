"""Visualization utilities for training and evaluation."""

import os
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional
import torch


class Visualizer:
    """Class for creating visualizations during training and evaluation."""

    def __init__(self, save_dir: str):
        """Initialize visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        best_losses: Optional[List[float]] = None,
        best_accs: Optional[List[float]] = None,
        save_name: str = "training_curves",
    ) -> np.ndarray:
        """Plot training curves for loss and accuracy.

        Args:
            train_losses: Training losses
            val_losses: Validation losses
            train_accs: Training accuracies
            val_accs: Validation accuracies
            best_losses: Best losses (optional)
            best_accs: Best accuracies (optional)
            save_name: Name for saved plots

        Returns:
            Combined plot as numpy array
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(train_losses, "b-", label="Train Loss", alpha=0.8)
        ax1.plot(val_losses, "g--", label="Val Loss", alpha=0.8)
        if best_losses:
            ax1.plot(best_losses, "r:", label="Best Loss", alpha=0.8)
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracies
        ax2.plot(train_accs, "b-", label="Train Acc", alpha=0.8)
        ax2.plot(val_accs, "g--", label="Val Acc", alpha=0.8)
        if best_accs:
            ax2.plot(best_accs, "r:", label="Best Acc", alpha=0.8)
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Convert to numpy array for display
        fig.canvas.draw()
        # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib versions
        try:
            plot_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            plot_array = plot_array[:, :, :3]
        except AttributeError:
            # Fallback for older matplotlib versions
            plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return plot_array

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        save_name: str = "confusion_matrix",
    ) -> np.ndarray:
        """Plot confusion matrix as heatmap.

        Args:
            confusion_matrix: Confusion matrix
            class_names: Class names for labels
            normalize: Whether to normalize the matrix
            save_name: Name for saved plot

        Returns:
            Heatmap image as numpy array
        """
        # Normalize if requested
        if normalize:
            cm = confusion_matrix.astype("float")
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm = cm / row_sums
            title = "Normalized Confusion Matrix"
            fmt = ".2f"
        else:
            cm = confusion_matrix.astype("int")
            title = "Confusion Matrix"
            fmt = "d"

        # Create heatmap using OpenCV for consistency with original code
        cm_normalized = cv2.normalize(
            cm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmap = cv2.applyColorMap(
            cv2.resize(
                cm_normalized, None, fx=50, fy=50, interpolation=cv2.INTER_NEAREST
            ),
            cv2.COLORMAP_HOT,
        )

        # Save using matplotlib for better annotations
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.hot)
        plt.title(title)
        plt.colorbar()

        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Also save the OpenCV version for compatibility
        cv2.imwrite(os.path.join(self.save_dir, f"{save_name}_heatmap.png"), heatmap)

        return heatmap

    def save_sample_images(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        save_name: str = "sample_images",
        max_samples: int = 16,
    ) -> None:
        """Save sample images with labels and predictions.

        Args:
            images: Image tensors (N, C, H, W)
            labels: True labels
            predictions: Predicted labels (optional)
            class_names: Class names for labels
            save_name: Name for saved plot
            max_samples: Maximum number of samples to show
        """
        batch_size = min(images.size(0), max_samples)
        grid_size = int(np.ceil(np.sqrt(batch_size)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(batch_size):
            # Convert tensor to numpy and transpose to HWC
            img = images[i].cpu().numpy().transpose(1, 2, 0)

            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)

            # Get labels
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                true_label = torch.argmax(labels[i]).item()
            else:
                true_label = labels[i].item()

            # Create title
            if class_names:
                title = f"True: {class_names[true_label]}"
                if predictions is not None:
                    pred_label = torch.argmax(predictions[i]).item()
                    title += f"\nPred: {class_names[pred_label]}"
            else:
                title = f"True: {true_label}"
                if predictions is not None:
                    pred_label = torch.argmax(predictions[i]).item()
                    title += f"\nPred: {pred_label}"

            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=8)
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def create_learning_rate_plot(
        self,
        learning_rates: List[float],
        losses: List[float],
        save_name: str = "learning_rate_plot",
    ) -> None:
        """Create learning rate vs loss plot for LR finding.

        Args:
            learning_rates: Learning rates
            losses: Corresponding losses
            save_name: Name for saved plot
        """
        plt.figure(figsize=(10, 6))
        plt.semilogx(learning_rates, losses)
        plt.title("Learning Rate vs Loss")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def numpy_to_cv2_display(
        img_array: np.ndarray, window_name: str = "Display", wait_key: int = 1
    ) -> None:
        """Display numpy array using OpenCV.

        Args:
            img_array: Image array (RGB format)
            window_name: Window name for display
            wait_key: Wait key duration
        """
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            display_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            display_img = img_array

        cv2.imshow(window_name, display_img)
        cv2.waitKey(wait_key)
