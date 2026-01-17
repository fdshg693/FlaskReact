from loguru import logger

from config import PROJECTPATHS
from ml.numeric.dataset import MLCompatibleDataset, MLDatasetConverter
from ml.numeric.pipeline import (
    execute_machine_learning_pipeline,
    store_model_and_learning_logs,
)

iris_data: MLCompatibleDataset = MLDatasetConverter.convert(
    PROJECTPATHS.iris_data_path,
    features=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
    target="target",
)


iris_model, iris_net, iris_acc_hist, iris_loss_hist, iris_exp_name = (
    execute_machine_learning_pipeline(dataset=iris_data, epochs=2)
)
store_model_and_learning_logs(
    trained_model=iris_net,
    accuracy_history=iris_acc_hist,
    loss_history=iris_loss_hist,
    dataset_name="iris",
    epochs=2,
    experiment_name=iris_exp_name,
)
logger.info(f"Iris test metric (acc) = {iris_model.evaluate_model():.4f}")
