"""Stock Closing Price Prediction"""
import argparse
import os

import pandas as pd
import tensorflow as tf

import autokeras as ak


def get_data(file_path, train_ratio):
    dataset = pd.read_csv(file_path, names=['price',])

    val_split = int(len(dataset) * train_ratio)
    data_train = dataset[:val_split]
    data_validation = dataset[val_split:]
    
    return dataset, data_train, data_validation


def get_data_x(data):
    return data[
        ['price']
    ].astype('float64')


def get_data_y(data):
    return data['price'].astype('float64')


def train(
    x_train, y_train, x_validation, y_validation,
    batch_size, epochs,
    lookback, max_trials
):
    clf = ak.TimeseriesForecaster(
        project_name='stock_prediction',
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=argparse.FileType('r'),
                        help='training file path, with closing prices one per line')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    # batch_size equalled to 32 will cause the "unbroadcastable error", not knowing why
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    # larget lookback may increase prediction accuracy
    # but will also decrease validation samples, causing missing of "val_loss"
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--max_trials', type=int, default=50)
    return parser.parse_args()


def main():
    args = get_args()

    dataset, data_train, data_validation = get_data(args.file_path, args.train_ratio)
    x_train, y_train = get_data_x(data_train), get_data_y(data_train)
    x_validation, y_validation = get_data_x(data_validation), get_data_y(data_validation)
    print(f'train data shapes: x {x_train.shape}, y {y_train.shape}')
    print(f'validation data shapes: x {x_validation.shape}, y {y_validation.shape}')
    clf = train(
        x_train=x_train, y_train=y_train,
        x_validation=x_validation, y_validation=y_validation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lookback=args.lookback,
        max_trials=args.max_trials
    )
    
    # Predict with the best model(includes original training data).
    predictions = clf.predict(get_data_x(dataset))
    print(predictions.shape)
    
    # Evaluate the best model with testing data.
    print(clf.evaluate(get_data_x(data_validation), get_data_y(data_validation)))


if __name__ == '__main__':
    main()
