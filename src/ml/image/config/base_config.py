"""YAML-based configuration management for experiments."""

import datetime
import os
from typing import Any, Dict, Optional

import yaml


class BaseConfig:
    """YAML-based configuration class with validation and path management."""

    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """Initialize configuration from dictionary data."""
        if config_data is None:
            config_data = {}

        # Store raw config data
        self._config_data = config_data.copy()

        # Set all config values as attributes
        for key, value in config_data.items():
            setattr(self, key, value)

        # Ensure required attributes exist with defaults
        self._set_defaults()

        # Run post-initialization setup
        self._post_init()

    def _set_defaults(self):
        """Set default values for required attributes."""
        defaults = {
            # Dataset settings
            "dataset_name": "Oketo_KIT3",
            "dataset_path": "./dataset/Oketo_KIT3/",
            "num_class": 0,
            "img_size": 128,
            "img_scale": 1.5,
            "aug_scale": 0.5,
            "aug_brightness": 50.0,
            # Model settings
            "layer": 3,
            "num_hidden": 4096,
            "l2softmax": True,
            "dropout_rate": 0.2,
            # Training settings
            "epoch": 12000,
            "batch_size_train": 10,
            "batch_size_test": 10,
            "learning_rate": 0.001,
            # Device settings
            "device": "cuda",
            # Output settings
            "log_dir": "./logs/",
            "checkpoint_dir": "./checkpoints/",
            # Experiment management
            "auto_timestamp": True,
            "auto_cuda_detection": True,
            "experiment_name_template": "{timestamp}_img{img_size}_layer{layer}_hidden{num_hidden}_{num_class}class_dropout{dropout_rate}_scale{img_scale}_{dataset_name}",
        }

        for key, default_value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, default_value)

    def _post_init(self):
        """Post-initialization to set up derived paths and settings."""
        # Auto CUDA detection
        if getattr(self, "auto_cuda_detection", True):
            self.device = (
                "cuda" if self.device == "cuda" and self._is_cuda_available() else "cpu"
            )

        # Handle dynamic dataset_path (f-string style)
        if hasattr(self, "dataset_path") and "{dataset_name}" in str(self.dataset_path):
            self.dataset_path = f"./dataset/{self.dataset_name}/"

        # Generate experiment name with timestamp
        if getattr(self, "auto_timestamp", True):
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            template = getattr(
                self,
                "experiment_name_template",
                "{timestamp}_img{img_size}_layer{layer}_hidden{num_hidden}_{num_class}class_dropout{dropout_rate}_scale{img_scale}_{dataset_name}",
            )

            self.experiment_name = template.format(
                timestamp=timestamp,
                img_size=self.img_size,
                layer=self.layer,
                num_hidden=self.num_hidden,
                num_class=self.num_class,
                dropout_rate=self.dropout_rate,
                img_scale=self.img_scale,
                dataset_name=self.dataset_name,
            )

            # Update log and checkpoint directories
            self.log_dir = os.path.join(self.log_dir, self.experiment_name)
            self.checkpoint_dir = os.path.join(
                self.checkpoint_dir, self.experiment_name
            )

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        # Create dictionary from all attributes that don't start with _
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_yaml(
        cls, yaml_path: str, base_config_path: Optional[str] = None
    ) -> "BaseConfig":
        """Load configuration from YAML file with optional base config inheritance."""

        # Load base configuration if specified
        base_config = {}
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_config = yaml.safe_load(f) or {}

        # Load experiment-specific configuration
        experiment_config = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                experiment_config = yaml.safe_load(f) or {}

        # Merge configurations (experiment overrides base)
        merged_config = {**base_config, **experiment_config}

        return cls(merged_config)

    @classmethod
    def from_default(cls) -> "BaseConfig":
        """Load configuration with default values only."""
        return cls({})

    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ConfigManager:
    """YAML-based configuration manager for experiments."""

    def __init__(
        self, config_path: Optional[str] = None, base_config_path: Optional[str] = None
    ):
        """Initialize configuration manager.

        Args:
            config_path: Path to experiment-specific configuration file.
            base_config_path: Path to base configuration file. If None, uses default_config.yaml.
        """
        # Set default base config path
        if base_config_path is None:
            base_config_path = os.path.join(
                os.path.dirname(__file__), "default_config.yaml"
            )

        if config_path and os.path.exists(config_path):
            self.config = BaseConfig.from_yaml(config_path, base_config_path)
        elif os.path.exists(base_config_path):
            self.config = BaseConfig.from_yaml(base_config_path)
        else:
            self.config = BaseConfig.from_default()

    def get_config(self) -> BaseConfig:
        """Get current configuration."""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            # Also update the internal config data
            self.config._config_data[key] = value

        # Re-run post_init to update derived paths
        self.config._post_init()

    def save_config(self, save_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file."""
        if save_path is None:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.log_dir), exist_ok=True)
            save_path = os.path.join(self.config.log_dir, "config.yaml")

        self.config.save_yaml(save_path)

    def load_experiment_config(self, experiment_name: str) -> None:
        """Load a specific experiment configuration by name."""
        experiment_path = os.path.join(
            os.path.dirname(__file__), "experiment_configs", f"{experiment_name}.yaml"
        )

        if os.path.exists(experiment_path):
            base_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")
            self.config = BaseConfig.from_yaml(experiment_path, base_path)
        else:
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
