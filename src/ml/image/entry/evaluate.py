"""Command-line interface and callable API for model evaluation.

This module now exposes a parameterized ``main()`` suitable for programmatic use
and a ``cli()`` wrapper that preserves the original command-line behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from loguru import logger

from ml.image.core.evaluation.model_evaluator import ModelEvaluator


def find_available_checkpoints(
    checkpoint_base_dir: str | Path = "./checkpoints/",
) -> list[dict[str, str]]:
    """Find all available model checkpoints.

    Args:
        checkpoint_base_dir: Base directory containing checkpoints

    Returns:
        List of available checkpoint path dicts
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
    """List all available checkpoints."""
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
    """Evaluate a model on a single image or a directory of images.

    Args:
        model: Path to model checkpoint (.pth)
        image: Path to single image file
        directory: Path to directory containing images
        output: Optional path to save results (JSON)
        visualize: If True, show/save visualization for single image
        device: Inference device ("cuda" or "cpu")
        img_size: Preprocessing image size
        list_checkpoints: If True, list checkpoints and return

    Returns:
        - For single-image: prediction result dict
        - For directory: summary dict
        - For list-checkpoints or on validation failure: None
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
