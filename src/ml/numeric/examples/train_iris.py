from loguru import logger
from ml.numeric.pipeline import (
    train_and_save_pipeline,
)
from ml.numeric.dataset import MLDatasetConverter
from sklearn.datasets import load_iris

print("=== Iris Classification Test ===")
iris = load_iris()
iris_ds = MLDatasetConverter.convert(iris)

# Use the new combined pipeline
iris_model, iris_net, iris_acc_hist, iris_loss_hist, iris_exp_name = (
    train_and_save_pipeline(dataset=iris_ds, dataset_name="iris", epochs=3)
)
logger.info(f"Iris test metric (acc) = {iris_model.evaluate_model():.4f}")
