from machineLearning.dataset import ml_dataset_from_csv, MLCompatibleDataset
from config import PATHS
from machineLearning.ml_class import (
    execute_machine_learning_pipeline,
    save_model_and_learning_curves_with_custom_name,
)
from loguru import logger

iris_data: MLCompatibleDataset = ml_dataset_from_csv(
    PATHS.iris_data_path,
    features=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
    target="target",
)


iris_model, iris_net, iris_acc_hist, iris_loss_hist = execute_machine_learning_pipeline(
    iris_data, epochs=2
)
save_model_and_learning_curves_with_custom_name(
    iris_net, iris_acc_hist, iris_loss_hist, "iris", epochs=2
)
logger.info(f"Iris test metric (acc) = {iris_model.evaluate_model():.4f}")
