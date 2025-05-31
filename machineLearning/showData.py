from sklearn.datasets import load_iris, load_diabetes
import os

iris_data = load_iris()
diabetes_data = load_diabetes()

# diabetes_dataをCSVにして保存
import pandas as pd

diabetes_df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
diabetes_df["target"] = diabetes_data.target
save_path = os.path.join(os.path.dirname(__file__), "../data/diabetes_data.csv")
diabetes_df.to_csv(save_path, index=False)
