"""Logging utilities for training and evaluation."""

import os
import csv
import json
import logging
import datetime
from typing import Dict, Any, List
import numpy as np


class Logger:
    """Custom logger for training experiments."""

    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """Initialize logger.

        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        # Initialize log files
        self.log_file = os.path.join(log_dir, "training.log")
        self.csv_file = os.path.join(log_dir, "metrics.csv")
        self.json_file = os.path.join(log_dir, "experiment_info.json")

        # Set up Python logger
        self._setup_logger()

        # Initialize CSV file
        self._initialize_csv()

        # Store metrics for CSV logging
        self.csv_data = []

    def _setup_logger(self) -> None:
        """Set up Python logger."""
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        if not os.path.exists(self.csv_file):
            headers = [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "best_loss",
                "best_acc",
                "learning_rate",
                "elapsed_time",
            ]
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Message to log
        """
        self.logger.error(message)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        best_loss: float,
        best_acc: float,
        learning_rate: float,
        elapsed_time: float,
    ) -> None:
        """Log epoch metrics.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            best_loss: Best loss so far
            best_acc: Best accuracy so far
            learning_rate: Current learning rate
            elapsed_time: Elapsed time for epoch
        """
        # Log to console/file
        message = (
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f} | "
            f"Best Loss: {best_loss:.6f} | Best Acc: {best_acc:.4f} | "
            f"LR: {learning_rate:.6f} | Time: {elapsed_time:.2f}s"
        )
        self.log_info(message)

        # Store for CSV
        row_data = [
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_loss,
            best_acc,
            learning_rate,
            elapsed_time,
        ]
        self.csv_data.append(row_data)

        # Write to CSV immediately for real-time monitoring
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information.

        Args:
            model_info: Model information dictionary
        """
        self.log_info("=" * 50)
        self.log_info("MODEL INFORMATION")
        self.log_info("=" * 50)

        for key, value in model_info.items():
            self.log_info(f"{key}: {value}")

        self.log_info("=" * 50)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.log_info("=" * 50)
        self.log_info("EXPERIMENT CONFIGURATION")
        self.log_info("=" * 50)

        for key, value in config.items():
            self.log_info(f"{key}: {value}")

        self.log_info("=" * 50)

        # Save to JSON for later reference
        with open(self.json_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Log detailed metrics.

        Args:
            metrics: Metrics dictionary
            prefix: Prefix for log messages
        """
        if prefix:
            self.log_info(f"--- {prefix} METRICS ---")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_info(f"{key}: {value:.6f}")
            elif isinstance(value, np.ndarray):
                if value.size <= 10:  # Only log small arrays
                    self.log_info(f"{key}: {value}")
                else:
                    self.log_info(f"{key}: array shape {value.shape}")
            else:
                self.log_info(f"{key}: {value}")

    def save_numpy_logs(
        self,
        train_losses: List[float],
        train_accs: List[float],
        val_losses: List[float],
        val_accs: List[float],
        best_losses: List[float],
        best_accs: List[float],
    ) -> None:
        """Save logs as numpy arrays for compatibility.

        Args:
            train_losses: Training losses
            train_accs: Training accuracies
            val_losses: Validation losses
            val_accs: Validation accuracies
            best_losses: Best losses
            best_accs: Best accuracies
        """
        log_array = np.vstack(
            [train_losses, train_accs, val_losses, val_accs, best_losses, best_accs]
        ).T

        numpy_log_file = os.path.join(self.log_dir, "log.csv")
        np.savetxt(numpy_log_file, log_array, delimiter=",", fmt="%.8f")

        self.log_info(f"Saved numpy logs to {numpy_log_file}")

    def create_summary(
        self,
        final_train_acc: float,
        final_val_acc: float,
        best_val_acc: float,
        total_epochs: int,
        total_time: float,
    ) -> None:
        """Create experiment summary.

        Args:
            final_train_acc: Final training accuracy
            final_val_acc: Final validation accuracy
            best_val_acc: Best validation accuracy achieved
            total_epochs: Total number of epochs
            total_time: Total training time
        """
        self.log_info("=" * 50)
        self.log_info("EXPERIMENT SUMMARY")
        self.log_info("=" * 50)
        self.log_info(f"Total Epochs: {total_epochs}")
        self.log_info(
            f"Total Training Time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)"
        )
        self.log_info(f"Final Training Accuracy: {final_train_acc:.4f}")
        self.log_info(f"Final Validation Accuracy: {final_val_acc:.4f}")
        self.log_info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        self.log_info(f"Time per Epoch: {total_time / total_epochs:.2f} seconds")
        self.log_info("=" * 50)

        # Save summary to JSON
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": total_epochs,
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "best_val_accuracy": best_val_acc,
            "time_per_epoch": total_time / total_epochs,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        """Close logger and clean up handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
