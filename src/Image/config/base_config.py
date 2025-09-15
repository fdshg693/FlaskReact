"""YAML-based configuration management for experiments.

This module provides a flexible configuration management system for machine learning
experiments. It supports YAML-based configuration files with inheritance, automatic
path management, and experiment tracking.

Classes:
    BaseConfig: Core configuration class with YAML support and automatic path generation.
    ConfigManager: High-level manager for configuration lifecycle management.

Example:
    >>> config = BaseConfig.from_yaml('experiment.yaml')
    >>> config.dataset_name = 'new_dataset'
    >>> config.save_yaml('updated_config.yaml')
"""
import os
import yaml
import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path


class BaseConfig:
    """YAML-based configuration class with validation and path management.
    
    This class provides a comprehensive configuration management system for machine
    learning experiments. It supports default values, automatic path generation,
    CUDA detection, and experiment naming with timestamps.
    
    Features:
        - Automatic default value setting for common ML parameters
        - Dynamic experiment name generation with timestamps
        - CUDA availability detection
        - Path management for logs and checkpoints
        - YAML serialization/deserialization
        - Configuration inheritance support
    
    Attributes:
        dataset_name (str): Name of the dataset being used
        dataset_path (str): Path to the dataset directory
        num_class (int): Number of classes in the dataset
        img_size (int): Input image size for the model
        img_scale (float): Scaling factor for image preprocessing
        layer (int): Number of layers in the model
        num_hidden (int): Number of hidden units
        dropout_rate (float): Dropout rate for regularization
        epoch (int): Number of training epochs
        batch_size_train (int): Training batch size
        batch_size_test (int): Testing batch size
        learning_rate (float): Learning rate for optimization
        device (str): Computing device ('cuda' or 'cpu')
        log_dir (str): Directory for logging outputs
        checkpoint_dir (str): Directory for model checkpoints
        experiment_name (str): Auto-generated experiment identifier
    
    Example:
        >>> # Create with defaults
        >>> config = BaseConfig.from_default()
        >>> print(config.device)  # 'cuda' or 'cpu' (auto-detected)
        
        >>> # Load from YAML
        >>> config = BaseConfig.from_yaml('config.yaml')
        >>> config.learning_rate = 0.01
        >>> config.save_yaml('updated_config.yaml')
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration from dictionary data.
        
        Args:
            config_data: Dictionary containing configuration parameters.
                        If None, an empty dictionary is used and defaults are applied.
        
        Note:
            After setting initial values, this method calls _set_defaults() to ensure
            all required attributes exist, then _post_init() for derived calculations.
        """
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
    
    
    def _set_defaults(self) -> None:
        """Set default values for required attributes.
        
        This method ensures all necessary configuration attributes exist with
        sensible default values. It covers dataset settings, model architecture,
        training parameters, device configuration, and experiment management.
        
        Default Categories:
            - Dataset: name, path, classes, image preprocessing
            - Model: architecture layers, hidden units, regularization
            - Training: epochs, batch sizes, learning rate
            - Device: CUDA/CPU selection
            - Output: logging and checkpoint directories
            - Experiment: naming templates and auto-generation flags
        
        Note:
            Only sets attributes that don't already exist, preserving any
            values that were explicitly provided during initialization.
        """
        defaults = {
            # Dataset settings
            'dataset_name': 'Oketo_KIT3',
            'dataset_path': './dataset/Oketo_KIT3/',
            'num_class': 0,
            'img_size': 128,
            'img_scale': 1.5,
            'aug_scale': 0.5,
            'aug_brightness': 50.0,
            
            # Model settings
            'layer': 3,
            'num_hidden': 4096,
            'l2softmax': True,
            'dropout_rate': 0.2,
            
            # Training settings
            'epoch': 12000,
            'batch_size_train': 10,
            'batch_size_test': 10,
            'learning_rate': 0.001,
            
            # Device settings
            'device': 'cuda',
            
            # Output settings
            'log_dir': './logs/',
            'checkpoint_dir': './checkpoints/',
            
            # Experiment management
            'auto_timestamp': True,
            'auto_cuda_detection': True,
            'experiment_name_template': '{timestamp}_img{img_size}_layer{layer}_hidden{num_hidden}_{num_class}class_dropout{dropout_rate}_scale{img_scale}_{dataset_name}'
        }
        
        for key, default_value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, default_value)
    
    def _post_init(self) -> None:
        """Post-initialization to set up derived paths and settings.
        
        This method handles dynamic configuration setup that depends on other
        configuration values. It performs:
        
        1. CUDA detection and device assignment
        2. Dynamic dataset path resolution using f-string templates
        3. Experiment name generation with timestamps
        4. Log and checkpoint directory path construction
        
        The method is called automatically after _set_defaults() and can be
        called manually after configuration updates to refresh derived values.
        
        Side Effects:
            - Updates self.device based on CUDA availability
            - Modifies self.dataset_path if it contains template variables
            - Generates new self.experiment_name with current timestamp
            - Updates self.log_dir and self.checkpoint_dir paths
        """
        # Auto CUDA detection
        if getattr(self, 'auto_cuda_detection', True):
            self.device = 'cuda' if self.device == 'cuda' and self._is_cuda_available() else 'cpu'
        
        # Handle dynamic dataset_path (f-string style)
        if hasattr(self, 'dataset_path') and '{dataset_name}' in str(self.dataset_path):
            self.dataset_path = f"./dataset/{self.dataset_name}/"
        
        # Generate experiment name with timestamp
        if getattr(self, 'auto_timestamp', True):
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            
            template = getattr(self, 'experiment_name_template', 
                             '{timestamp}_img{img_size}_layer{layer}_hidden{num_hidden}_{num_class}class_dropout{dropout_rate}_scale{img_scale}_{dataset_name}')
            
            self.experiment_name = template.format(
                timestamp=timestamp,
                img_size=self.img_size,
                layer=self.layer,
                num_hidden=self.num_hidden,
                num_class=self.num_class,
                dropout_rate=self.dropout_rate,
                img_scale=self.img_scale,
                dataset_name=self.dataset_name
            )
            
            # Update log and checkpoint directories
            self.log_dir = os.path.join(self.log_dir, self.experiment_name)
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.experiment_name)

    
    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available for PyTorch operations.
        
        Returns:
            bool: True if CUDA is available and PyTorch can use it, False otherwise.
                 Also returns False if PyTorch is not installed.
        
        Note:
            This method gracefully handles the case where PyTorch is not installed
            by catching ImportError and returning False.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary containing all public configuration attributes.
                           Private attributes (those starting with '_') are excluded.
        
        Note:
            This method is useful for serialization to YAML or JSON formats,
            and for creating configuration snapshots.
        """
        # Create dictionary from all attributes that don't start with _
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_yaml(cls, yaml_path: str, base_config_path: Optional[str] = None) -> 'BaseConfig':
        """Load configuration from YAML file with optional base config inheritance.
        
        This method supports configuration inheritance where an experiment-specific
        configuration can inherit from a base configuration file. Settings in the
        experiment file override those in the base file.
        
        Args:
            yaml_path: Path to the experiment-specific YAML configuration file.
            base_config_path: Optional path to base configuration file. If provided
                            and the file exists, its settings are loaded first.
        
        Returns:
            BaseConfig: New configuration instance with merged settings.
        
        Example:
            >>> # Load with inheritance
            >>> config = BaseConfig.from_yaml('experiment.yaml', 'base.yaml')
            
            >>> # Load single file
            >>> config = BaseConfig.from_yaml('config.yaml')
        
        Note:
            If either YAML file doesn't exist, it's treated as an empty configuration.
            The method uses yaml.safe_load for security.
        """
        
        # Load base configuration if specified
        base_config = {}
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_config = yaml.safe_load(f) or {}
        
        # Load experiment-specific configuration
        experiment_config = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                experiment_config = yaml.safe_load(f) or {}
        
        # Merge configurations (experiment overrides base)
        merged_config = {**base_config, **experiment_config}
        
        return cls(merged_config)
    
    @classmethod
    def from_default(cls) -> 'BaseConfig':
        """Load configuration with default values only.
        
        Returns:
            BaseConfig: New configuration instance using only default values
                       defined in _set_defaults().
        
        Example:
            >>> config = BaseConfig.from_default()
            >>> print(config.learning_rate)  # 0.001 (default value)
        """
        return cls({})
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path where the YAML configuration file will be saved.
                      Parent directories are created if they don't exist.
        
        Raises:
            OSError: If the file cannot be created or written to.
        
        Example:
            >>> config = BaseConfig.from_default()
            >>> config.learning_rate = 0.01
            >>> config.save_yaml('experiments/my_config.yaml')
        
        Note:
            The output YAML is formatted with default_flow_style=False for
            better readability.
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ConfigManager:
    """YAML-based configuration manager for experiments.
    
    This class provides high-level management of configuration lifecycle including
    loading, updating, saving, and experiment-specific configuration handling.
    It acts as a wrapper around BaseConfig with additional convenience methods.
    
    Features:
        - Automatic base configuration loading
        - Configuration updates with derived path recalculation
        - Experiment configuration loading by name
        - Automatic save path generation
    
    Attributes:
        config (BaseConfig): The managed configuration instance.
    
    Example:
        >>> manager = ConfigManager('experiment.yaml')
        >>> manager.update_config(learning_rate=0.01, batch_size_train=32)
        >>> manager.save_config()  # Saves to auto-generated path
        
        >>> # Load predefined experiment
        >>> manager.load_experiment_config('resnet_baseline')
    """
    
    def __init__(self, config_path: Optional[str] = None, base_config_path: Optional[str] = None) -> None:
        """Initialize configuration manager.
        
        Args:
            config_path: Path to experiment-specific configuration file.
                        If None or file doesn't exist, falls back to base config.
            base_config_path: Path to base configuration file. If None, 
                            uses 'default_config.yaml' in the same directory as this module.
        
        Note:
            The initialization follows this priority:
            1. If config_path exists: load it with base_config inheritance
            2. If only base_config exists: load base_config alone
            3. Otherwise: create configuration with defaults only
        """
        # Set default base config path
        if base_config_path is None:
            base_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        
        if config_path and os.path.exists(config_path):
            self.config = BaseConfig.from_yaml(config_path, base_config_path)
        elif os.path.exists(base_config_path):
            self.config = BaseConfig.from_yaml(base_config_path)
        else:
            self.config = BaseConfig.from_default()
    
    def get_config(self) -> BaseConfig:
        """Get current configuration instance.
        
        Returns:
            BaseConfig: The currently managed configuration object.
        
        Example:
            >>> manager = ConfigManager()
            >>> config = manager.get_config()
            >>> print(config.dataset_name)
        """
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values.
        
        This method updates the configuration attributes and automatically
        recalculates derived paths and settings by calling _post_init().
        
        Args:
            **kwargs: Arbitrary keyword arguments representing configuration
                     parameters to update.
        
        Example:
            >>> manager = ConfigManager()
            >>> manager.update_config(
            ...     learning_rate=0.01,
            ...     batch_size_train=64,
            ...     dataset_name='CIFAR10'
            ... )
        
        Note:
            After updating, derived paths (log_dir, checkpoint_dir) and
            experiment names are automatically recalculated.
        """
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            # Also update the internal config data
            self.config._config_data[key] = value
        
        # Re-run post_init to update derived paths
        self.config._post_init()
    
    def save_config(self, save_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file.
        
        Args:
            save_path: Optional custom path for saving. If None, automatically
                      generates path as '{log_dir}/config.yaml'.
        
        Example:
            >>> manager = ConfigManager()
            >>> manager.save_config()  # Saves to auto-generated path
            >>> manager.save_config('custom/path/config.yaml')  # Custom path
        
        Note:
            When using auto-generated path, the log directory is created
            if it doesn't exist.
        """
        if save_path is None:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.log_dir), exist_ok=True)
            save_path = os.path.join(self.config.log_dir, 'config.yaml')
        
        self.config.save_yaml(save_path)
    
    def load_experiment_config(self, experiment_name: str) -> None:
        """Load a specific experiment configuration by name.
        
        This method loads predefined experiment configurations stored in the
        'experiment_configs' directory. It uses inheritance from the default
        base configuration.
        
        Args:
            experiment_name: Name of the experiment configuration file 
                           (without .yaml extension).
        
        Raises:
            FileNotFoundError: If the experiment configuration file doesn't exist.
        
        Example:
            >>> manager = ConfigManager()
            >>> manager.load_experiment_config('resnet_baseline')
            >>> manager.load_experiment_config('mobilenet_small')
        
        Note:
            Experiment configs should be stored in:
            '{module_dir}/experiment_configs/{experiment_name}.yaml'
        """
        experiment_path = os.path.join(
            os.path.dirname(__file__), 
            'experiment_configs', 
            f'{experiment_name}.yaml'
        )
        
        if os.path.exists(experiment_path):
            base_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
            self.config = BaseConfig.from_yaml(experiment_path, base_path)
        else:
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")
