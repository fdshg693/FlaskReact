"""Model evaluation utilities for wood classification."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from ml.image.core.models.image_model import ImageModel


class ModelEvaluator:
    """Evaluator for trained wood classification models."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", img_size: int = 128):
        """Initialize model evaluator.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run evaluation on ('cuda' or 'cpu')
            img_size: Image size for preprocessing
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # Load model and configuration
        self.model = None
        self.class_names = None
        self.num_classes = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {self.checkpoint_path}")

        # Load checkpoint with weights_only=False for compatibility
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Extract model configuration from checkpoint info
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")

        model_config = {}

        # Try to get config from checkpoint_info.json
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
                model_config = info.get("model_config", {})

        # Try to get config from the checkpoint itself
        if not model_config and "model_config" in checkpoint:
            model_config = checkpoint["model_config"]

        # Extract from filename if no config found
        if not model_config:
            filename = os.path.basename(checkpoint_dir)
            model_config = self._parse_config_from_filename(filename)

        # Auto-detect number of classes from dataset if not specified
        if model_config.get("num_class", 0) == 0:
            # Try to detect from dataset path in filename
            if "test_dataset" in checkpoint_dir.lower():
                # Count classes in test_dataset directory
                dataset_path = "./dataset/test_dataset/"
                if os.path.exists(dataset_path):
                    class_dirs = [
                        d
                        for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))
                    ]
                    model_config["num_class"] = len(class_dirs)
                    print(
                        f"Auto-detected {len(class_dirs)} classes from dataset: {class_dirs}"
                    )
                else:
                    model_config["num_class"] = 3  # Default for test_dataset
            else:
                model_config["num_class"] = 2  # Default fallback

        # Initialize model with configuration
        self.num_classes = model_config.get("num_class", 3)

        print(f"Model configuration: {model_config}")
        print(f"Number of classes: {self.num_classes}")
        self.model = ImageModel(
            num_class=self.num_classes,
            img_size=model_config.get("img_size", self.img_size),
            layer=model_config.get("layer", 3),
            num_hidden=model_config.get("num_hidden", 4096),
            l2softmax=model_config.get("l2softmax", True),
            dropout_rate=model_config.get("dropout_rate", 0.2),
        )

        # Load model weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the checkpoint is the state_dict itself
            self.model.load_state_dict(checkpoint)

        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")
        print(f"Model info: {self.model.get_model_info()}")

    def _parse_config_from_filename(self, filename: str) -> Dict[str, Any]:
        """Parse model configuration from checkpoint directory name.

        Args:
            filename: Checkpoint directory name

        Returns:
            Parsed configuration dictionary
        """
        config = {}
        parts = filename.split("_")

        for part in parts:
            if part.startswith("img") and part[3:].isdigit():
                config["img_size"] = int(part[3:])
            elif part.startswith("layer") and part[5:].isdigit():
                config["layer"] = int(part[5:])
            elif part.startswith("hidden") and part[6:].isdigit():
                config["num_hidden"] = int(part[6:])
            elif part.endswith("class"):
                class_num_str = part[:-5]
                if class_num_str.isdigit():
                    num_class = int(class_num_str)
                    # If it's 0 class and dataset is test_dataset, set to 3
                    if num_class == 0 and "test_dataset" in filename:
                        config["num_class"] = 3
                    else:
                        config["num_class"] = num_class
            elif part.startswith("dropout") and len(part) > 7:
                try:
                    config["dropout_rate"] = float(part[7:])
                except ValueError:
                    pass
            elif part.startswith("scale") and len(part) > 5:
                try:
                    config["img_scale"] = float(part[5:])
                except ValueError:
                    pass

        return config

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image for evaluation.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Predict class for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)

        return self.predict_image_data(img_tensor)

    def predict_image_data(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs.tolist(),
            "raw_output": output[0].cpu().numpy().tolist(),
        }

        return result

    def predict_batch_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict classes for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of prediction results
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
                print(
                    f"Processed: {os.path.basename(image_path)} -> Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}"
                )
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({"image_path": image_path, "error": str(e)})

        return results

    def evaluate_directory(
        self, image_dir: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate all images in a directory.

        Args:
            image_dir: Directory containing images to evaluate
            output_file: Optional path to save results as JSON

        Returns:
            Evaluation results summary
        """
        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

        image_paths = [str(p) for p in image_paths]

        if not image_paths:
            raise ValueError(f"No images found in directory: {image_dir}")

        print(f"Found {len(image_paths)} images to evaluate")

        # Predict all images
        results = self.predict_batch_images(image_paths)

        # Calculate summary statistics
        successful_predictions = [r for r in results if "error" not in r]
        failed_predictions = [r for r in results if "error" in r]

        summary = {
            "total_images": len(image_paths),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(failed_predictions),
            "results": results,
        }

        if successful_predictions:
            # Class distribution
            class_counts = {}
            confidence_scores = []

            for result in successful_predictions:
                pred_class = result["predicted_class"]
                confidence = result["confidence"]

                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                confidence_scores.append(confidence)

            summary["class_distribution"] = class_counts
            summary["average_confidence"] = np.mean(confidence_scores)
            summary["min_confidence"] = np.min(confidence_scores)
            summary["max_confidence"] = np.max(confidence_scores)

        # Save results if output file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to: {output_file}")

        return summary

    def visualize_prediction(
        self, image_path: str, save_path: Optional[str] = None, show_image: bool = True
    ) -> np.ndarray:
        """Visualize prediction result with image and probabilities.

        Args:
            image_path: Path to the image file
            save_path: Optional path to save visualization
            show_image: Whether to display the image

        Returns:
            Visualization image as numpy array
        """
        # Get prediction
        result = self.predict_single_image(image_path)

        # Load and prepare image for visualization
        img = cv2.imread(image_path)
        img = cv2.resize(img, (400, 400))

        # Create visualization
        vis_height = 600
        vis_width = 600
        vis_img = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255

        # Place image
        img_start_y = 20
        img_end_y = img_start_y + 400
        img_start_x = (vis_width - 400) // 2
        img_end_x = img_start_x + 400

        vis_img[img_start_y:img_end_y, img_start_x:img_end_x] = img

        # Add prediction text
        text_y = img_end_y + 30

        # Predicted class
        pred_text = f"Predicted Class: {result['predicted_class']}"
        cv2.putText(
            vis_img,
            pred_text,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Confidence
        conf_text = f"Confidence: {result['confidence']:.4f}"
        cv2.putText(
            vis_img,
            conf_text,
            (20, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Top probabilities
        probs = result["probabilities"]
        sorted_indices = np.argsort(probs)[::-1]  # Sort in descending order

        prob_text_y = text_y + 70
        cv2.putText(
            vis_img,
            "Class Probabilities:",
            (20, prob_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        for i, class_idx in enumerate(sorted_indices[:5]):  # Show top 5
            prob_value = probs[class_idx]
            prob_text = f"Class {class_idx}: {prob_value:.4f}"
            cv2.putText(
                vis_img,
                prob_text,
                (30, prob_text_y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )

        # Save visualization
        if save_path:
            cv2.imwrite(save_path, vis_img)
            print(f"Visualization saved to: {save_path}")

        # Show image
        if show_image:
            cv2.imshow("Prediction Result", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return vis_img


def predict_image_data(
    checkpoint_path: str, img_tensor: torch.Tensor
) -> Dict[str, Any]:
    """Predict class for a given image tensor using the specified model checkpoint.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        img_tensor: Preprocessed image tensor

    Returns:
        Dictionary containing prediction results
    """
    evaluator = ModelEvaluator(checkpoint_path=checkpoint_path)
    return evaluator.predict_image_data(img_tensor)
