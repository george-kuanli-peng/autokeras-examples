"""California Housing

Adapted from [Structured Data Regression](https://autokeras.com/tutorial/structured_data_regression/)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

import autokeras as ak


# Prepare data
house_dataset = fetch_california_housing()

df = pd.DataFrame(
    np.concatenate(
        (house_dataset.data, house_dataset.target.reshape(-1, 1)), axis=1
    ),
    columns=house_dataset.feature_names + ["Price"],
)

train_size = int(df.shape[0] * 0.9)
train_file_path = "train.csv"
test_file_path = "eval.csv"
df[:train_size].to_csv(train_file_path, index=False)
df[train_size:].to_csv(test_file_path, index=False)

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)  # It tries 3 different models.

# Feed the structured data regressor with training data.
reg.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    "Price",
    epochs=10,
)

# Predict with the best model.
predicted_y = reg.predict(test_file_path)

# Evaluate the best model with testing data.
print(reg.evaluate(test_file_path, "Price"))
