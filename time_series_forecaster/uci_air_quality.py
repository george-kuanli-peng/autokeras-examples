"""UCI Air Quality Forcasting

Adapted from [TimeSeriesForecaster](https://autokeras.com/tutorial/timeseries_forecaster/)"""
import os

import pandas as pd
import tensorflow as tf

import autokeras as ak


def get_data(train_ratio=0.7):
    dataset_file_path = tf.keras.utils.get_file(
        fname="AirQualityUCI.zip",
        origin="https://archive.ics.uci.edu/ml/machine-learning-databases/00360/"
        "AirQualityUCI.zip",
        extract=True,
        archive_format='zip'
    )
    dataset_file_path = os.path.join(os.path.dirname(dataset_file_path), 'AirQualityUCI.csv')

    dataset = pd.read_csv(dataset_file_path, sep=';')
    dataset = dataset[dataset.columns[:-2]].dropna().replace(',', '.', regex=True)

    val_split = int(len(dataset) * train_ratio)
    data_train = dataset[:val_split]
    data_validation = dataset[val_split:]
    
    return dataset, data_train, data_validation


def get_data_x(data):
    return data[
        [
            "CO(GT)",
            "PT08.S1(CO)",
            "NMHC(GT)",
            "C6H6(GT)",
            "PT08.S2(NMHC)",
            "NOx(GT)",
            "PT08.S3(NOx)",
            "NO2(GT)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
        ]
    ].astype('float64')


def get_data_y(data):
        return data["AH"].astype("float64")

 
def train(
    x_train, y_train, x_validation, y_validation,
    batch_size, epochs,
    lookback, max_trials
):
    clf = ak.TimeseriesForecaster(
        lookback=lookback,
        max_trials=max_trials,
        objective='val_loss'
    )
    clf.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_validation, y_validation),
        batch_size=batch_size,
        epochs=epochs
    )
    return clf


def main():
    dataset, data_train, data_validation = get_data()
    
    data_x_train = get_data_x(data_train)
    data_x_validation = get_data_x(data_validation)
    data_x_test = get_data_x(dataset)
    
    data_y_train = get_data_y(data_train)
    data_y_validation = get_data_y(data_validation)
    
    print(data_x_train.shape)
    print(data_y_train.shape)
    
    clf = train(
        x_train=data_x_train, y_train=data_y_train,
        x_validation=data_x_validation, y_validation=data_y_validation,
        batch_size=32, epochs=10,
        lookback=3, max_trials=10
    )
    
    # Predict with the best model(includes original training data).
    predictions = clf.predict(data_x_test)
    print(predictions.shape)
    
    # Evaluate the best model with testing data.
    print(clf.evaluate(data_x_validation, data_y_validation))


if __name__ == '__main__':
    main()
