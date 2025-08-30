"""Convenience script to run model evaluation."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluate_model import main

if __name__ == "__main__":
    main()
