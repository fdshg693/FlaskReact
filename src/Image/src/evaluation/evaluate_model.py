"""Interactive GUI for model evaluation."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from pathlib import Path
import json
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluator import ModelEvaluator


class EvaluationGUI:
    """GUI for model evaluation."""
    
    def __init__(self, root):
        """Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Wood Classification Model Evaluator")
        self.root.geometry("800x600")
        
        self.evaluator = None
        self.current_image_path = None
        self.current_result = None
        
        self.setup_ui()
        self.load_available_checkpoints()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Model selection section
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="5")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Checkpoint:").grid(row=0, column=0, sticky=tk.W)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(model_frame, textvariable=self.checkpoint_var, 
                                           state="readonly", width=50)
        self.checkpoint_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=2, padx=(5, 0))
        
        ttk.Button(model_frame, text="Browse...", 
                  command=self.browse_checkpoint).grid(row=0, column=3, padx=(5, 0))
        
        # Status
        self.status_var = tk.StringVar(value="No model loaded")
        ttk.Label(model_frame, textvariable=self.status_var, 
                 foreground="red").grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # Image selection section
        image_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="5")
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        image_frame.columnconfigure(1, weight=1)
        
        ttk.Button(image_frame, text="Select Image", 
                  command=self.select_image).grid(row=0, column=0)
        
        self.image_path_var = tk.StringVar(value="No image selected")
        ttk.Label(image_frame, textvariable=self.image_path_var).grid(row=0, column=1, 
                                                                     sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Button(image_frame, text="Evaluate", 
                  command=self.evaluate_image).grid(row=0, column=2, padx=(5, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Create notebook for results tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image")
        
        self.image_label = ttk.Label(self.image_frame, text="No image loaded")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Prediction tab
        self.pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_frame, text="Prediction")
        
        # Create text widget for prediction results
        pred_text_frame = ttk.Frame(self.pred_frame)
        pred_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.pred_text = tk.Text(pred_text_frame, wrap=tk.WORD, state=tk.DISABLED)
        pred_scrollbar = ttk.Scrollbar(pred_text_frame, orient=tk.VERTICAL, command=self.pred_text.yview)
        self.pred_text.configure(yscrollcommand=pred_scrollbar.set)
        
        self.pred_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pred_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export button
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(export_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT)
        
        ttk.Button(export_frame, text="Batch Evaluate Directory", 
                  command=self.batch_evaluate).pack(side=tk.LEFT, padx=(10, 0))
    
    def load_available_checkpoints(self):
        """Load available checkpoints into the combobox."""
        checkpoints = []
        checkpoint_base_dir = "./checkpoints/"
        
        if os.path.exists(checkpoint_base_dir):
            for exp_dir in os.listdir(checkpoint_base_dir):
                exp_path = os.path.join(checkpoint_base_dir, exp_dir)
                if os.path.isdir(exp_path):
                    # Look for common checkpoint files
                    checkpoint_files = ['best_accuracy.pth', 'best_loss.pth', 'latest.pth']
                    for checkpoint_file in checkpoint_files:
                        checkpoint_path = os.path.join(exp_path, checkpoint_file)
                        if os.path.exists(checkpoint_path):
                            display_name = f"{exp_dir} - {checkpoint_file}"
                            checkpoints.append((display_name, checkpoint_path))
        
        self.checkpoint_combo['values'] = [item[0] for item in checkpoints]
        self.checkpoint_paths = {item[0]: item[1] for item in checkpoints}
        
        if checkpoints:
            self.checkpoint_combo.current(0)
    
    def browse_checkpoint(self):
        """Browse for checkpoint file."""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            display_name = f"Custom - {os.path.basename(file_path)}"
            self.checkpoint_paths[display_name] = file_path
            current_values = list(self.checkpoint_combo['values'])
            current_values.append(display_name)
            self.checkpoint_combo['values'] = current_values
            self.checkpoint_var.set(display_name)
    
    def load_model(self):
        """Load the selected model."""
        selected = self.checkpoint_var.get()
        if not selected or selected not in self.checkpoint_paths:
            messagebox.showerror("Error", "Please select a checkpoint")
            return
        
        checkpoint_path = self.checkpoint_paths[selected]
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            self.evaluator = ModelEvaluator(checkpoint_path)
            self.status_var.set(f"Model loaded successfully ({self.evaluator.num_classes} classes)")
            self.root.configure(fg="green")
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            self.evaluator = None
            messagebox.showerror("Error", f"Failed to load model:\\n{str(e)}")
    
    def select_image(self):
        """Select an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_path_var.set(os.path.basename(file_path))
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Display the selected image."""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            
            # Calculate size to fit in the display area
            display_size = (400, 400)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {str(e)}", image="")
            self.image_label.image = None
    
    def evaluate_image(self):
        """Evaluate the selected image."""
        if not self.evaluator:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.current_image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            # Run evaluation in a separate thread to prevent GUI freezing
            def run_evaluation():
                self.current_result = self.evaluator.predict_single_image(self.current_image_path)
                self.root.after(0, self.display_results)
            
            self.pred_text.configure(state=tk.NORMAL)
            self.pred_text.delete(1.0, tk.END)
            self.pred_text.insert(tk.END, "Evaluating...")
            self.pred_text.configure(state=tk.DISABLED)
            
            thread = threading.Thread(target=run_evaluation)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed:\\n{str(e)}")
    
    def display_results(self):
        """Display evaluation results."""
        if not self.current_result:
            return
        
        result = self.current_result
        
        # Format results text
        results_text = f"""Image: {os.path.basename(result['image_path'])}

Prediction Results:
==================
Predicted Class: {result['predicted_class']}
Confidence: {result['confidence']:.4f}

Class Probabilities:
===================
"""
        
        for i, prob in enumerate(result['probabilities']):
            results_text += f"Class {i}: {prob:.4f}\\n"
        
        results_text += f"""

Raw Output Values:
==================
"""
        for i, val in enumerate(result['raw_output']):
            results_text += f"Output {i}: {val:.4f}\\n"
        
        # Update text widget
        self.pred_text.configure(state=tk.NORMAL)
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, results_text)
        self.pred_text.configure(state=tk.DISABLED)
        
        # Switch to prediction tab
        self.notebook.select(self.pred_frame)
    
    def export_results(self):
        """Export current results to JSON."""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_result, f, indent=2)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\\n{str(e)}")
    
    def batch_evaluate(self):
        """Evaluate all images in a directory."""
        if not self.evaluator:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if not directory:
            return
        
        output_file = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        def run_batch_evaluation():
            try:
                # Update status
                self.root.after(0, lambda: self.status_var.set("Running batch evaluation..."))
                
                # Run evaluation
                summary = self.evaluator.evaluate_directory(directory, output_file)
                
                # Show results
                self.root.after(0, lambda: self.show_batch_results(summary, output_file))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch evaluation failed:\\n{str(e)}"))
            finally:
                self.root.after(0, lambda: self.status_var.set(f"Model loaded successfully ({self.evaluator.num_classes} classes)"))
        
        thread = threading.Thread(target=run_batch_evaluation)
        thread.daemon = True
        thread.start()
    
    def show_batch_results(self, summary, output_file):
        """Show batch evaluation results."""
        message = f"""Batch Evaluation Complete!

Results saved to: {output_file}

Summary:
- Total images: {summary['total_images']}
- Successful predictions: {summary['successful_predictions']}
- Failed predictions: {summary['failed_predictions']}
"""
        
        if summary['successful_predictions'] > 0:
            message += f"""
- Average confidence: {summary['average_confidence']:.4f}
- Confidence range: {summary['min_confidence']:.4f} - {summary['max_confidence']:.4f}

Class Distribution:
"""
            for class_id, count in summary['class_distribution'].items():
                percentage = (count / summary['successful_predictions']) * 100
                message += f"- Class {class_id}: {count} images ({percentage:.1f}%)\\n"
        
        messagebox.showinfo("Batch Evaluation Results", message)


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = EvaluationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
