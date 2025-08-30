"""Command-line interface for model evaluation."""
import argparse
import os
import sys
from pathlib import Path
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from image.core.evaluation.evaluator import ModelEvaluator


def find_available_checkpoints(checkpoint_base_dir: str = "./checkpoints/") -> list:
    """Find all available model checkpoints.
    
    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        
    Returns:
        List of available checkpoint paths
    """
    checkpoints = []
    
    if not os.path.exists(checkpoint_base_dir):
        return checkpoints
    
    for exp_dir in os.listdir(checkpoint_base_dir):
        exp_path = os.path.join(checkpoint_base_dir, exp_dir)
        if os.path.isdir(exp_path):
            # Look for common checkpoint files
            checkpoint_files = ['best_accuracy.pth', 'best_loss.pth', 'latest.pth']
            for checkpoint_file in checkpoint_files:
                checkpoint_path = os.path.join(exp_path, checkpoint_file)
                if os.path.exists(checkpoint_path):
                    checkpoints.append({
                        'experiment': exp_dir,
                        'checkpoint': checkpoint_file,
                        'path': checkpoint_path
                    })
    
    return checkpoints


def list_checkpoints():
    """List all available checkpoints."""
    checkpoints = find_available_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found in ./checkpoints/ directory")
        return
    
    print("Available checkpoints:")
    print("-" * 80)
    
    current_exp = None
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint['experiment'] != current_exp:
            current_exp = checkpoint['experiment']
            print(f"\nExperiment: {current_exp}")
        
        print(f"  [{i}] {checkpoint['checkpoint']} -> {checkpoint['path']}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate wood classification model')
    
    # Model selection
    parser.add_argument('--model', '-m', type=str,
                       help='Path to model checkpoint (.pth file)')
    
    # Input specification
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image', '-i', type=str,
                      help='Path to single image file')
    group.add_argument('--directory', '-d', type=str,
                      help='Path to directory containing images')
    
    # Options
    parser.add_argument('--output', '-o', type=str,
                       help='Output file to save results (JSON format)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show visualization of predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--img-size', type=int, default=128,
                       help='Image size for preprocessing (default: 128)')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='List all available checkpoints and exit')
    
    args = parser.parse_args()
    
    # List checkpoints and exit if requested
    if args.list_checkpoints:
        list_checkpoints()
        return
    
    # Check required arguments when not listing checkpoints
    if not args.model:
        print("Error: --model/-m is required when not using --list-checkpoints")
        print("Use --list-checkpoints to see available checkpoints")
        return
    
    if not args.image and not args.directory:
        print("Error: Either --image/-i or --directory/-d is required")
        return
    
    # Validate model checkpoint path
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        print("\nUse --list-checkpoints to see available checkpoints")
        return
    
    # Validate input path
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if args.directory and not os.path.exists(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        return
    
    try:
        # Initialize evaluator
        print(f"Initializing evaluator with model: {args.model}")
        evaluator = ModelEvaluator(
            checkpoint_path=args.model,
            device=args.device,
            img_size=args.img_size
        )
        
        # Single image evaluation
        if args.image:
            print(f"\\nEvaluating single image: {args.image}")
            result = evaluator.predict_single_image(args.image)
            
            # Print results
            print("\\nPrediction Results:")
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print("  Class Probabilities:")
            for i, prob in enumerate(result['probabilities']):
                print(f"    Class {i}: {prob:.4f}")
            
            # Visualize if requested
            if args.visualize:
                vis_path = None
                if args.output:
                    base_name = os.path.splitext(args.output)[0]
                    vis_path = f"{base_name}_visualization.jpg"
                
                evaluator.visualize_prediction(
                    args.image, 
                    save_path=vis_path,
                    show_image=True
                )
            
            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\\nResults saved to: {args.output}")
        
        # Directory evaluation
        elif args.directory:
            print(f"\\nEvaluating directory: {args.directory}")
            summary = evaluator.evaluate_directory(args.directory, args.output)
            
            # Print summary
            print("\\nEvaluation Summary:")
            print(f"  Total images: {summary['total_images']}")
            print(f"  Successful predictions: {summary['successful_predictions']}")
            print(f"  Failed predictions: {summary['failed_predictions']}")
            
            if summary['successful_predictions'] > 0:
                print(f"  Average confidence: {summary['average_confidence']:.4f}")
                print(f"  Confidence range: {summary['min_confidence']:.4f} - {summary['max_confidence']:.4f}")
                print("  Class distribution:")
                for class_id, count in summary['class_distribution'].items():
                    percentage = (count / summary['successful_predictions']) * 100
                    print(f"    Class {class_id}: {count} images ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
