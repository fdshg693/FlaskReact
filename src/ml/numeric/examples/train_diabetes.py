from loguru import logger
from ml.numeric.pipeline import (
    train_and_save_pipeline,
)
from ml.numeric.dataset import MLDatasetConverter

if __name__ == "__main__":
    # =============================Iris=============================

    # =============================Diabetes=============================
    from sklearn.datasets import load_diabetes

    print("=== Diabetes Regression Test ===")
    diabetes = load_diabetes()
    diab_ds = MLDatasetConverter.convert(diabetes)

    # Use the new combined pipeline
    diab_model, diab_net, diab_r2_hist, diab_loss_hist, diab_exp_name = (
        train_and_save_pipeline(dataset=diab_ds, dataset_name="diabetes", epochs=3)
    )
    logger.info(f"Diabetes test metric (R2) = {diab_model.evaluate_model():.4f}")
