"""Model evaluation utilities for wood classification."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from image.core.models.wood_net import WoodNet


class ModelEvaluator:
    """Class for evaluating and inferencing wood classification models.
    
    This class loads a trained WoodNet model and provides functionality for
    single image prediction, batch image prediction, and directory-wide evaluation.
    
    Attributes:
        checkpoint_path (str): Path to the trained model checkpoint file
        device (str): Device for inference execution ('cuda' or 'cpu')
        img_size (int): Image size for preprocessing
        model (Optional[WoodNet]): Loaded model instance
        class_names (Optional[List[str]]): List of class names
        num_classes (Optional[int]): Number of classes
        
    Example:
        >>> evaluator = ModelEvaluator(
        ...     checkpoint_path="./checkpoints/best_model.pth",
        ...     device="cuda",
        ...     img_size=128
        ... )
        >>> result = evaluator.predict_single_image("test_image.jpg")
        >>> print(f"Predicted class: {result['predicted_class']}")
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", img_size: int = 128) -> None:
        """Initialize ModelEvaluator.
        
        Loads the model from checkpoint and sets up evaluation configuration.
        Model configuration is automatically inferred from checkpoint information or filename.

        Args:
            checkpoint_path (str): Path to the trained model checkpoint file (.pth format)
            device (str, optional): Device for inference execution. Specify 'cuda' or 'cpu'.
                Defaults to 'cuda', but automatically changes to 'cpu' if CUDA is not available.
            img_size (int, optional): Image resize size for preprocessing (square).
                Defaults to 128.
                
        Raises:
            FileNotFoundError: If the checkpoint file does not exist
            RuntimeError: If model loading fails
            
        Note:
            Model configuration (number of classes, layers, etc.) is determined in the following priority:
            1. model_config in checkpoint_info.json
            2. model_config in checkpoint
            3. Parsing from directory name
            4. Default values
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # Load model and configuration
        self.model = None
        self.class_names = None
        self.num_classes = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from checkpoint.
        
        This internal method performs the following processes:
        1. Load checkpoint file
        2. Get model configuration (from checkpoint_info.json, checkpoint, or filename)
        3. Auto-detect number of classes
        4. Initialize WoodNet model
        5. Load trained weights
        6. Set to evaluation mode
        
        Raises:
            FileNotFoundError: If checkpoint file is not found
            json.JSONDecodeError: If checkpoint_info.json format is invalid
            KeyError: If model state dict keys are not found
            RuntimeError: If model loading fails
            
        Note:
            If number of classes is 0, attempts auto-detection from dataset directory.
            Defaults to 3 classes for test_dataset, 2 classes for others.
        """
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
        saved_class_names = []

        # Try to get config from checkpoint_info.json
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
                model_config = info.get("model_config", {})
                saved_class_names = info.get("class_names", [])
                if saved_class_names:
                    print(f"Found saved class names: {saved_class_names}")

        # Try to get config from the checkpoint itself
        if not model_config and "model_config" in checkpoint:
            model_config = checkpoint["model_config"]

        # Extract from filename if no config found
        if not model_config:
            filename = os.path.basename(checkpoint_dir)
            model_config = self._parse_config_from_filename(filename)

        # Auto-detect number of classes from dataset if not specified
        if model_config.get("num_class", 0) == 0:
            # 保存されたクラス名がある場合はそれを使用
            if saved_class_names:
                model_config["num_class"] = len(saved_class_names)
                self.class_names = saved_class_names
                print(f"Using saved class names: {saved_class_names}")
            else:
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
                        # ディレクトリ名をソートしてOS依存を回避
                        class_dirs.sort()
                        model_config["num_class"] = len(class_dirs)
                        self.class_names = class_dirs
                        print(
                            f"Auto-detected {len(class_dirs)} classes from dataset: {class_dirs}"
                        )
                    else:
                        model_config["num_class"] = 3  # Default for test_dataset
                        self.class_names = [f"class_{i}" for i in range(3)]
                else:
                    model_config["num_class"] = 2  # Default fallback
                    self.class_names = [f"class_{i}" for i in range(2)]
        else:
            # クラス数が指定されている場合、保存されたクラス名があれば使用
            if saved_class_names:
                self.class_names = saved_class_names
            else:
                # デフォルトのクラス名を生成
                self.class_names = [f"class_{i}" for i in range(model_config.get("num_class", 3))]

        # Initialize model with configuration
        self.num_classes = model_config.get("num_class", 3)

        print(f"Model configuration: {model_config}")
        print(f"Number of classes: {self.num_classes}")
        self.model = WoodNet(
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

        Extracts model hyperparameters based on directory naming conventions.
        Example: "2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset"

        Args:
            filename (str): Checkpoint directory name
                Format: [timestamp_]img{size}_layer{num}_hidden{size}_{num}class_dropout{rate}_scale{factor}_{dataset}

        Returns:
            Dict[str, Any]: Parsed model configuration dictionary
                - img_size (int): Image size
                - layer (int): Number of network layers
                - num_hidden (int): Number of hidden units
                - num_class (int): Number of classes
                - dropout_rate (float): Dropout rate
                - img_scale (float): Image scale factor
                
        Example:
            >>> evaluator = ModelEvaluator("path/to/checkpoint.pth")
            >>> config = evaluator._parse_config_from_filename(
            ...     "img128_layer3_hidden4096_3class_dropout0.2_test_dataset"
            ... )
            >>> print(config)
            {'img_size': 128, 'layer': 3, 'num_hidden': 4096, 'num_class': 3, 'dropout_rate': 0.2}
            
        Note:
            - 0class with test_dataset is automatically changed to 3class
            - Invalid values are ignored and default values are used
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
        """Preprocess a single image for evaluation.

        Loads an image file and converts it to the format suitable for model input.
        Processing steps:
        1. Load image file (OpenCV BGR format)
        2. Convert BGR → RGB color space
        3. Resize to specified size
        4. Normalize to [0, 1] range
        5. Convert to PyTorch tensor and add batch dimension

        Args:
            image_path (str): Path to the image file to process
                Supported formats: .jpg, .jpeg, .png, .bmp, .tiff

        Returns:
            torch.Tensor: Preprocessed image tensor
                Shape: (1, 3, img_size, img_size)
                Data type: float32
                Value range: [0.0, 1.0]
                
        Raises:
            ValueError: If the image file cannot be loaded
            FileNotFoundError: If the image file does not exist
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth", img_size=128)
            >>> tensor = evaluator.preprocess_image("sample.jpg")
            >>> print(tensor.shape)  # torch.Size([1, 3, 128, 128])
            
        Note:
            - Images are resized to square format, which may change aspect ratio
            - If GPU is available, tensor is automatically placed on GPU memory
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
        """Execute class prediction for a single image.

        Preprocesses the specified image file and performs classification prediction
        with the trained model. Calculates probability distribution using softmax
        function and uses the highest probability class as the prediction result.

        Args:
            image_path (str): Path to the image file for prediction
                Supported formats: .jpg, .jpeg, .png, .bmp, .tiff

        Returns:
            Dict[str, Any]: Dictionary containing prediction results
                - predicted_class (int): Index of predicted class (0-based)
                - predicted_class_name (str): Name of predicted class
                - confidence (float): Confidence of predicted class (0.0-1.0)
                - probabilities (List[float]): Prediction probability list for all classes
                - raw_output (List[float]): Model raw output (logit values)
                
        Raises:
            ValueError: If the image file cannot be loaded
            FileNotFoundError: If the image file does not exist
            RuntimeError: If an error occurs during model inference
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> result = evaluator.predict_single_image("wood_sample.jpg")
            >>> print(f"Predicted class: {result['predicted_class']}")
            >>> print(f"Confidence: {result['confidence']:.3f}")
            >>> for i, prob in enumerate(result['probabilities']):
            ...     print(f"Class {i}: {prob:.3f}")
            
        Note:
            - Prediction is executed in model evaluation mode with gradient computation disabled
            - Confidence is the softmax probability value of the predicted class
            - raw_output contains logit values before activation function application
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)

        return self.predict_image_data(img_tensor)

    def predict_image_data(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """Execute class prediction for preprocessed image tensor.

        Performs model inference directly on preprocessed PyTorch tensor.
        This method is efficient when image data is already in tensor format.

        Args:
            img_tensor (torch.Tensor): Preprocessed image tensor
                Expected shape: (batch_size, 3, height, width)
                Data type: float32
                Value range: [0.0, 1.0]

        Returns:
            Dict[str, Any]: Dictionary containing prediction results
                - predicted_class (int): Index of predicted class (0-based)
                - predicted_class_name (str): Name of predicted class
                - confidence (float): Confidence of predicted class (0.0-1.0)
                - probabilities (List[float]): Prediction probability list for all classes
                - raw_output (List[float]): Model raw output (logit values)
                
        Raises:
            RuntimeError: If an error occurs during model inference
            ValueError: If tensor shape or data type is invalid
            
        Example:
            >>> import torch
            >>> evaluator = ModelEvaluator("model.pth")
            >>> # For preprocessed tensor
            >>> tensor = torch.randn(1, 3, 128, 128)
            >>> result = evaluator.predict_image_data(tensor)
            >>> print(f"Predicted class: {result['predicted_class']}")
            
        Note:
            - If batch size is greater than 1, only the result of the first sample is returned
            - Inference is executed without gradient computation for memory efficiency
            - Tensor must be placed on the appropriate device (CPU/GPU)
        """
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()

        # Get class name for the predicted class
        predicted_class_name = self.get_class_name_by_index(predicted_class)

        result = {
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "probabilities": all_probs.tolist(),
            "raw_output": output[0].cpu().numpy().tolist(),
        }

        return result

    def get_class_names(self) -> List[str]:
        """保存されたクラス名のリストを取得します。
        
        Returns:
            List[str]: クラス名のリスト。順序はデータセット作成時の順番と同じ。
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> class_names = evaluator.get_class_names()
            >>> print(f"Available classes: {class_names}")
            >>> result = evaluator.predict_single_image("test.jpg")
            >>> predicted_class_name = class_names[result['predicted_class']]
            >>> print(f"Predicted class name: {predicted_class_name}")
        """
        return self.class_names if self.class_names else []

    def get_class_name_by_index(self, class_index: int) -> str:
        """クラスインデックスからクラス名を取得します。
        
        Args:
            class_index (int): クラスのインデックス（0ベース）
            
        Returns:
            str: クラス名。インデックスが無効な場合はデフォルト名を返す。
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> result = evaluator.predict_single_image("test.jpg")
            >>> class_name = evaluator.get_class_name_by_index(result['predicted_class'])
            >>> print(f"Predicted: {class_name}")
        """
        if self.class_names and 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        else:
            return f"class_{class_index}"

    def predict_batch_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Execute class prediction for multiple images in batch.

        Executes prediction sequentially for multiple specified image files
        and returns prediction results as a list. For images where errors occur,
        dictionaries containing error information are also included in the results.

        Args:
            image_paths (List[str]): List of image file paths for prediction
                Each path must be a valid image file format

        Returns:
            List[Dict[str, Any]]: List of prediction results for each image
                Dictionary format for successful cases:
                - predicted_class (int): Predicted class
                - predicted_class_name (str): Name of predicted class
                - confidence (float): Confidence
                - probabilities (List[float]): Class probabilities
                - raw_output (List[float]): Raw output
                
                Dictionary format for failed cases:
                - image_path (str): Image path where error occurred
                - error (str): Error message
                
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> paths = ["img1.jpg", "img2.jpg", "invalid.txt"]
            >>> results = evaluator.predict_batch_images(paths)
            >>> for i, result in enumerate(results):
            ...     if "error" in result:
            ...         print(f"Image {i}: Error - {result['error']}")
            ...     else:
            ...         print(f"Image {i}: Class {result['predicted_class']}")
                
        Note:
            - Each image is processed independently, so processing continues
              for other images even if errors occur in some images
            - Processing progress is output to the console
            - Be careful about memory usage when processing large numbers of images
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
        """Execute batch evaluation for all images in a directory.

        Automatically detects image files in the specified directory,
        executes prediction for all images, and summarizes statistical information.
        Results can optionally be saved as a JSON file.

        Args:
            image_dir (str): Directory path containing images to evaluate
            output_file (Optional[str], optional): JSON file path for saving results
                If not specified, results are not saved

        Returns:
            Dict[str, Any]: Evaluation results summary dictionary
                - total_images (int): Total number of images processed
                - successful_predictions (int): Number of successful predictions
                - failed_predictions (int): Number of failed predictions
                - results (List[Dict]): Detailed results for each image
                - class_distribution (Dict[int, int]): Number of images per class
                - average_confidence (float): Average confidence
                - min_confidence (float): Minimum confidence
                - max_confidence (float): Maximum confidence
                
        Raises:
            ValueError: If no image files are found in the directory
            FileNotFoundError: If the specified directory does not exist
            PermissionError: If there are no access permissions to directory or files
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> summary = evaluator.evaluate_directory(
            ...     "./test_images/",
            ...     output_file="evaluation_results.json"
            ... )
            >>> print(f"Success rate: {summary['successful_predictions']/summary['total_images']:.2%}")
            >>> print(f"Average confidence: {summary['average_confidence']:.3f}")
            
        Note:
            - Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff (case insensitive)
            - If the directory for the result JSON file does not exist, it is automatically created
            - Be careful about processing time and memory usage when processing large numbers of images
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
        """Visualize prediction results with image and probability information.

        Visually displays prediction results for the specified image and generates
        a visualization image including predicted class, confidence, and probability
        distribution for each class.

        Args:
            image_path (str): Path to the image file for visualization
            save_path (Optional[str], optional): Path to save visualization results
                If not specified, results are not saved
            show_image (bool, optional): Whether to display visualization results on screen
                Defaults to True

        Returns:
            np.ndarray: Visualization image array data
                Shape: (height, width, 3)
                Data type: uint8
                Color space: BGR (OpenCV format)
                
        Raises:
            ValueError: If the image file cannot be loaded
            FileNotFoundError: If the image file does not exist
            RuntimeError: If an error occurs during prediction processing
            
        Example:
            >>> evaluator = ModelEvaluator("model.pth")
            >>> vis_img = evaluator.visualize_prediction(
            ...     "sample.jpg",
            ...     save_path="result_visualization.png",
            ...     show_image=False
            ... )
            >>> print(f"Visualization image size: {vis_img.shape}")
            
        Note:
            - Visualization image is generated at 600x600 pixels
            - Original image is resized to 400x400 pixels and placed in the center
            - Probabilities for top 5 classes are displayed
            - If show_image=True, display ends with ESC key or window close
            - Directory for save path is not automatically created if it doesn't exist
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


def predict_image_with_checkpoint(
    checkpoint_path: str, img_tensor: torch.Tensor
) -> Dict[str, Any]:
    """Execute prediction for image tensor using the specified model checkpoint.

    This function serves as a convenience function for the ModelEvaluator class,
    creating a temporary model evaluator instance to execute prediction.
    It is suitable for one-time prediction processing, but for multiple predictions,
    using ModelEvaluator instance directly is more efficient.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint file
            (.pth format file)
        img_tensor (torch.Tensor): Preprocessed image tensor
            Expected shape: (batch_size, 3, height, width)
            Data type: float32, value range: [0.0, 1.0]

    Returns:
        Dict[str, Any]: Dictionary containing prediction results
            - predicted_class (int): Index of predicted class (0-based)
            - predicted_class_name (str): Name of predicted class
            - confidence (float): Confidence of predicted class (0.0-1.0)
            - probabilities (List[float]): Prediction probability list for all classes
            - raw_output (List[float]): Model raw output (logit values)
            
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If an error occurs during model loading or inference
        ValueError: If tensor shape or data type is invalid
        
    Example:
        >>> import torch
        >>> img_tensor = torch.randn(1, 3, 128, 128)
        >>> result = predict_with_checkpoint("./models/best_model.pth", img_tensor)
        >>> print(f"Predicted class: {result['predicted_class']}")
        >>> print(f"Confidence: {result['confidence']:.3f}")
        
    Note:
        - This function creates a ModelEvaluator instance internally, so
          overhead occurs when called multiple times
        - For frequent prediction processing, direct use of ModelEvaluator class is recommended
        - Device (CPU/GPU) is automatically detected and configured
    """
    evaluator = ModelEvaluator(checkpoint_path=checkpoint_path)
    return evaluator.predict_image_data(img_tensor)
