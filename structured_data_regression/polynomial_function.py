"""Polynomial Function Regression

y(x) := 2x^2 - x + 4.9 + noise
"""
import random

import autokeras as ak
import numpy as np
import pandas as pd
import tensorflow as tf


def gen_data(size, min_val=0.0, max_val=1.0E+3, max_noise=1.0):
    random.seed()
    data = {'x': [random.uniform(min_val, max_val) for _ in range(size)]}
    data['y'] = [
        2*(x**2) - x + 4.9 + random.uniform(0.0, max_noise)
        for x in data['x']
    ]
    return pd.DataFrame(data)


def main():
    data = gen_data(size=10**5)

    train_size = int(data.shape[0] * 0.8)
    data_train = data[:train_size]
    data_test = data[train_size:]

    reg = ak.StructuredDataRegressor(
        project_name='polynomial_function',
        overwrite=False, max_trials=50
    )
    reg.fit(
        x=data_train.loc[:, ['x']],
        y=data_train.loc[:, ['y']],
        epochs=10
    )
    y_pred = reg.predict(data_test.loc[:, ['x']])
    print(reg.evaluate(
        x=data_test.loc[:, ['x']],
        y=data_test.loc[:, ['y']]
    ))


if __name__ == '__main__':
    main()
