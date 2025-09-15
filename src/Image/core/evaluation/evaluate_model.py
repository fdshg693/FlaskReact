"""Comprehensive model evaluation command-line interface and programmatic API.

This module provides a complete evaluation framework for PyTorch image classification
models, supporting both interactive command-line usage and programmatic integration.
The implementation follows modern Python practices using pathlib, type hints,
loguru logging, and structured error handling.

The module exposes three primary functions:
- find_available_checkpoints(): Discovers available model checkpoints
- list_checkpoints(): Provides human-readable checkpoint listings  
- main(): Unified evaluation interface supporting multiple operational modes

Key Features:
    - Cross-platform path handling using pathlib.Path
    - Automatic CUDA/CPU device fallback
    - Comprehensive result logging and visualization
    - Batch directory evaluation with detailed statistics
    - Single image prediction with confidence scoring
    - JSON output format for integration with other tools
    - Graceful error handling with informative messages

Typical Usage:
    Command-line evaluation:
        $ python -m image.core.evaluation.evaluate_model \\
            --model ./checkpoints/best_model.pth \\
            --image ./test.jpg \\
            --visualize

    Programmatic usage:
        >>> from image.core.evaluation.evaluate_model import main
        >>> result = main(
        ...     model="./checkpoints/best_model.pth",
        ...     image="./test.jpg",
        ...     visualize=True
        ... )
        >>> print(f"Predicted: {result['predicted_class']}")

Dependencies:
    - torch: PyTorch model loading and inference
    - pathlib: Cross-platform file system operations
    - loguru: Structured logging with appropriate levels
    - image.core.evaluation.evaluator: Core evaluation engine

See Also:
    - ModelEvaluator: Core evaluation implementation
    - CLI module: Command-line argument parsing wrapper
"""

from __future__ import annotations
import json

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from loguru import logger
from image.core.evaluation.evaluator import ModelEvaluator


def find_available_checkpoints(
    checkpoint_base_dir: str | Path = "./checkpoints/",
) -> list[dict[str, str]]:
    """Discover and catalog available model checkpoints from experiment directories.

    Recursively searches the specified base directory for experiment subdirectories
    and identifies standard PyTorch checkpoint files. This function is designed to
    work with the project's checkpoint storage convention where each experiment
    creates a timestamped directory containing training artifacts.

    The search targets three types of checkpoint files commonly saved during training:
    - best_accuracy.pth: Checkpoint with highest validation accuracy
    - best_loss.pth: Checkpoint with lowest validation loss  
    - latest.pth: Most recent checkpoint (for resuming training)

    Args:
        checkpoint_base_dir (str | Path, optional): Absolute or relative path to the
            base directory containing experiment subdirectories. The function uses
            pathlib.Path for robust cross-platform path handling. 
            Defaults to "./checkpoints/".

    Returns:
        list[dict[str, str]]: Ordered list of checkpoint metadata dictionaries.
            Each dictionary contains exactly three keys:
            - 'experiment' (str): Name of the experiment directory (typically timestamp-based)
            - 'checkpoint' (str): Filename of the checkpoint file
            - 'path' (str): Absolute path to the checkpoint file for loading
            
            Returns empty list if base directory doesn't exist or contains no checkpoints.

    Example:
        >>> # Standard usage with default checkpoint directory
        >>> checkpoints = find_available_checkpoints()
        >>> len(checkpoints)
        6
        >>> checkpoints[0]
        {
            'experiment': '2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset',
            'checkpoint': 'best_accuracy.pth',
            'path': './checkpoints/2025_09_06.../best_accuracy.pth'
        }
        
        >>> # Custom checkpoint directory
        >>> custom_checkpoints = find_available_checkpoints("/path/to/experiments")
        >>> for cp in custom_checkpoints:
        ...     print(f"Found: {cp['experiment']}/{cp['checkpoint']}")
        Found: experiment_20250906/best_accuracy.pth
        Found: experiment_20250906/latest.pth
        Found: experiment_20250907/best_loss.pth

    Note:
        - Uses pathlib.Path for cross-platform compatibility
        - Non-existent base directories return empty list (no exceptions raised)
        - Only processes subdirectories, ignoring files in the base directory
        - Multiple checkpoint types in same experiment create separate list entries
        - Maintains discovery order for consistent checkpoint enumeration
        - Compatible with project's experiment naming convention: timestamp_configuration_dataset

    Raises:
        OSError: If filesystem access permissions prevent directory traversal
        
    See Also:
        - list_checkpoints(): Human-readable formatting of discovered checkpoints
        - ModelEvaluator.__init__(): Loads checkpoints discovered by this function
    """
    base = Path(checkpoint_base_dir)
    checkpoints: list[dict[str, str]] = []

    if not base.exists():
        return checkpoints

    for exp_path in base.iterdir():
        if exp_path.is_dir():
            # Look for common checkpoint files
            for checkpoint_file in ("best_accuracy.pth", "best_loss.pth", "latest.pth"):
                checkpoint_path = exp_path / checkpoint_file
                if checkpoint_path.exists():
                    checkpoints.append(
                        {
                            "experiment": exp_path.name,
                            "checkpoint": checkpoint_file,
                            "path": str(checkpoint_path),
                        }
                    )

    return checkpoints


def list_checkpoints() -> None:
    """Display comprehensive overview of all available model checkpoints.
    
    Provides a formatted, human-readable listing of discovered model checkpoints
    organized by experiment directory. This function serves as a command-line utility
    for users to identify and select appropriate checkpoints for evaluation.

    The output format groups checkpoints by experiment for clarity, with each
    checkpoint receiving a sequential index number for easy command-line reference.
    This indexing system integrates with command-line argument parsing workflows.

    Output Format:
        - Header section with total checkpoint count
        - Experiment groupings with full directory names  
        - Indexed checkpoint entries showing filename and full path
        - Separator lines for visual organization

    Side Effects:
        - Writes formatted output to logger.info() stream
        - No filesystem modifications or checkpoint validation
        - Exits silently if no checkpoints found

    Example Output:
        ```
        Available checkpoints:
        --------------------------------------------------------------------------------
        
        Experiment: 2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset
          [0] best_accuracy.pth -> ./checkpoints/2025_09_06.../best_accuracy.pth
          [1] best_loss.pth -> ./checkpoints/2025_09_06.../best_loss.pth
          [2] latest.pth -> ./checkpoints/2025_09_06.../latest.pth
        
        Experiment: 2025_09_07_20_15_30_img256_layer5_hidden2048_dropout0.3_scale2.0_validation_dataset
          [3] best_accuracy.pth -> ./checkpoints/2025_09_07.../best_accuracy.pth
          [4] latest.pth -> ./checkpoints/2025_09_07.../latest.pth
        ```

    Implementation Details:
        - Uses loguru logger for consistent output formatting
        - Maintains checkpoint discovery order for reproducible indexing
        - Groups by experiment directory while preserving global enumeration
        - Handles empty checkpoint directories gracefully
        - Compatible with both interactive and programmatic usage

    Note:
        - Indexes start from 0 and continue sequentially across experiments
        - No validation of checkpoint file integrity or compatibility
        - Output designed for human consumption; use find_available_checkpoints() for programmatic access
        - Experiment names preserve full timestamp and configuration details for uniqueness

    See Also:
        - find_available_checkpoints(): Underlying discovery mechanism
        - main(): Command-line interface that uses this function
        - ModelEvaluator: Loads checkpoints identified through this listing
    """
    checkpoints = find_available_checkpoints()

    if not checkpoints:
        logger.info("No checkpoints found in ./checkpoints/ directory")
        return

    logger.info("Available checkpoints:")
    logger.info("-" * 80)

    current_exp: Optional[str] = None
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint["experiment"] != current_exp:
            current_exp = checkpoint["experiment"]
            logger.info(f"\nExperiment: {current_exp}")

        logger.info(f"  [{i}] {checkpoint['checkpoint']} -> {checkpoint['path']}")


def main(
    *,
    model: str | Path | None = None,
    image: str | Path | None = None,
    directory: str | Path | None = None,
    output: str | Path | None = None,
    visualize: bool = False,
    device: Literal["cuda", "cpu"] = "cuda",
    img_size: int = 128,
    list_checkpoints: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute comprehensive model evaluation with flexible input and output options.

    This function serves as the primary interface for all model evaluation workflows,
    supporting both programmatic usage and command-line integration. It provides
    three distinct operational modes: checkpoint discovery, single-image prediction,
    and batch directory evaluation.

    The function implements a robust validation pipeline, graceful error handling,
    and comprehensive result logging. All file operations use pathlib.Path for
    cross-platform compatibility, and device selection automatically falls back
    to CPU when CUDA is unavailable.

    Operational Modes:
        1. Checkpoint Discovery: Lists available model checkpoints for user selection
        2. Single Image: Predicts class for individual image with optional visualization  
        3. Batch Directory: Evaluates all supported images in directory with statistics

    Args:
        model (str | Path | None, optional): Path to PyTorch model checkpoint (.pth file).
            Must exist and be loadable by ModelEvaluator. Required unless list_checkpoints=True.
            Supports both absolute and relative paths.
            
        image (str | Path | None, optional): Path to single image file for inference.
            Supports common formats (JPG, PNG, etc.). Mutually exclusive with directory.
            When specified, enables single-image prediction mode.
            
        directory (str | Path | None, optional): Path to directory containing images
            for batch evaluation. All supported image files will be processed.
            Mutually exclusive with image parameter. Enables batch evaluation mode.
            
        output (str | Path | None, optional): Path for saving evaluation results as JSON.
            Parent directories are created automatically if they don't exist.
            - Single image mode: Saves prediction results with confidence scores
            - Directory mode: Saves comprehensive evaluation summary with statistics
            
        visualize (bool, optional): Enable visualization output for single images.
            When True, generates and displays prediction visualization showing
            input image, predicted class, and confidence. Only applicable in
            single-image mode. Defaults to False.
            
        device (Literal["cuda", "cpu"], optional): Inference device specification.
            Automatically falls back to CPU if CUDA unavailable. ModelEvaluator
            handles device compatibility checks. Defaults to "cuda".
            
        img_size (int, optional): Target image size for preprocessing.
            Images are resized to (img_size, img_size) squares while maintaining
            aspect ratio through padding. Must match training configuration.
            Defaults to 128.
            
        list_checkpoints (bool, optional): Enable checkpoint discovery mode.
            When True, displays all available checkpoints and returns None.
            Overrides all other parameters. Defaults to False.

    Returns:
        Optional[Dict[str, Any]]: Evaluation results or None based on operational mode.
        
        Single Image Mode Returns:
            Dict containing:
            - 'predicted_class' (int): Predicted class index
            - 'confidence' (float): Maximum probability (0.0-1.0)
            - 'probabilities' (List[float]): Per-class probability distribution
            - 'raw_output' (List[float]): Raw model logits before softmax
            
        Directory Mode Returns:
            Dict containing:
            - 'total_images' (int): Total images processed
            - 'successful_predictions' (int): Successfully classified images
            - 'failed_predictions' (int): Images that failed processing
            - 'average_confidence' (float): Mean confidence across successful predictions
            - 'min_confidence' (float): Minimum confidence score
            - 'max_confidence' (float): Maximum confidence score
            - 'class_distribution' (Dict[str, int]): Count of predictions per class
            - 'detailed_results' (List[Dict]): Per-image results for successful predictions
            - 'failed_files' (List[str]): Filenames that failed processing
            
        Checkpoint Discovery Mode Returns:
            None (results displayed via logger)

    Raises:
        FileNotFoundError: When specified model, image, or directory paths don't exist
        ValueError: When required arguments are missing or mutually exclusive args provided
        RuntimeError: When model loading fails or device initialization fails
        OSError: When file system access permissions prevent operation
        
        All exceptions are caught and logged with detailed error messages,
        returning None for graceful failure handling.

    Example:
        >>> # Single image prediction with visualization
        >>> result = main(
        ...     model="./checkpoints/best_model.pth",
        ...     image="./test_image.jpg",
        ...     visualize=True,
        ...     output="./prediction.json"
        ... )
        >>> if result:
        ...     print(f"Predicted class: {result['predicted_class']}")
        ...     print(f"Confidence: {result['confidence']:.2%}")
        
        >>> # Batch directory evaluation with custom device
        >>> summary = main(
        ...     model="./checkpoints/latest.pth", 
        ...     directory="./validation_set/",
        ...     output="./batch_results.json",
        ...     device="cpu",
        ...     img_size=256
        ... )
        >>> if summary:
        ...     accuracy = summary['successful_predictions'] / summary['total_images']
        ...     print(f"Batch accuracy: {accuracy:.2%}")
        
        >>> # Discover available checkpoints
        >>> main(list_checkpoints=True)
        Available checkpoints:
        [0] experiment_20250906/best_accuracy.pth
        [1] experiment_20250906/latest.pth

    Implementation Details:
        - Uses pathlib.Path for all file system operations
        - Implements comprehensive path validation before evaluation
        - Leverages loguru for structured logging with appropriate log levels
        - Supports graceful degradation when CUDA unavailable
        - Creates output directories automatically using parents=True
        - Maintains backward compatibility with original CLI interface
        - Thread-safe for concurrent evaluation workflows

    Performance Considerations:
        - Single image: Optimized for interactive usage with immediate feedback
        - Directory batch: Progress logging for long-running evaluations
        - Memory efficient: Processes images individually in batch mode
        - GPU memory: Automatically manages device placement and cleanup

    See Also:
        - ModelEvaluator: Core evaluation engine used by this function
        - find_available_checkpoints(): Checkpoint discovery mechanism
        - list_checkpoints(): Human-readable checkpoint formatting
    """
    # List checkpoints and exit if requested
    if list_checkpoints:
        list_checkpoints_fn = list_checkpoints  # avoid shadowing in type checkers
        list_checkpoints_fn()
        return None

    # Check required arguments when not listing checkpoints
    if model is None:
        logger.error("--model/-m is required when not using --list-checkpoints")
        logger.info("Use --list-checkpoints to see available checkpoints")
        return None

    if image is None and directory is None:
        logger.error("Either --image/-i or --directory/-d is required")
        return None

    model_path = Path(model)
    image_path = Path(image) if image is not None else None
    directory_path = Path(directory) if directory is not None else None
    output_path = Path(output) if output is not None else None

    # Validate paths
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        logger.info("Use --list-checkpoints to see available checkpoints")
        return None

    if image_path is not None and not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return None

    if directory_path is not None and not directory_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        return None

    try:
        # Initialize evaluator
        logger.info(f"Initializing evaluator with model: {model_path}")
        evaluator = ModelEvaluator(
            checkpoint_path=str(model_path), device=device, img_size=img_size
        )

        # Single image evaluation
        if image_path is not None:
            logger.info(f"Evaluating single image: {image_path}")
            result: Dict[str, Any] = evaluator.predict_single_image(str(image_path))

            # Log results
            logger.info("Prediction Results:")
            logger.info(f"  Predicted Class: {result['predicted_class']}")
            logger.info(f"  Confidence: {result['confidence']:.4f}")
            logger.info("  Class Probabilities:")
            for i, prob in enumerate(result["probabilities"]):
                logger.info(f"    Class {i}: {prob:.4f}")

            # Visualize if requested
            if visualize:
                vis_path: Optional[str] = None
                if output_path is not None:
                    vis_path = str(
                        output_path.with_suffix("").as_posix() + "_visualization.jpg"
                    )

                evaluator.visualize_prediction(
                    str(image_path), save_path=vis_path, show_image=True
                )

            # Save results if output specified
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to: {output_path}")

            return result

        # Directory evaluation
        if directory_path is not None:
            logger.info(f"Evaluating directory: {directory_path}")
            summary: Dict[str, Any] = evaluator.evaluate_directory(
                str(directory_path),
                str(output_path) if output_path is not None else None,
            )

            # Log summary
            logger.info("Evaluation Summary:")
            logger.info(f"  Total images: {summary['total_images']}")
            logger.info(
                f"  Successful predictions: {summary['successful_predictions']}"
            )
            logger.info(f"  Failed predictions: {summary['failed_predictions']}")

            if summary["successful_predictions"] > 0:
                logger.info(
                    f"  Average confidence: {summary['average_confidence']:.4f}"
                )
                logger.info(
                    f"  Confidence range: {summary['min_confidence']:.4f} - {summary['max_confidence']:.4f}"
                )
                logger.info("  Class distribution:")
                for class_id, count in summary["class_distribution"].items():
                    percentage = (count / summary["successful_predictions"]) * 100
                    logger.info(
                        f"    Class {class_id}: {count} images ({percentage:.1f}%)"
                    )

            return summary

        return None

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Error during evaluation: {str(e)}")
        return None


if __name__ == "__main__":
    main(
        model="/Users/seiwan/CodeStudy/FlaskReact/checkpoints/2025_09_06_20_21_44_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset/best_accuracy.pth",
        image="/Users/seiwan/CodeStudy/FlaskReact/data/machineLearning/image/test_dataset/birch/image_01.jpg",
        directory=None,
        output=None,
        visualize=False,
        device="cuda",
        img_size=128,
        list_checkpoints=False,
    )
