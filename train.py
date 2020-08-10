"""
Train the NN model.
"""
import numpy as np
import pandas as pd
import warnings
from data.data import process_data
from model.model import get_gru
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, X_valid, y_valid, config):
    """
    train NN
    :param model:  Model, NN model to train.
    :param X_train: ndarray(number, timestep, feature), Input data for train.
    :param y_train: ndarray(number, ), result data for train.
    :param X_valid: ndarray(number, timestep, feature), Input data for valid.
    :param y_valid: ndarray(number, ), result data for valid.
    :param config: Dict, parameter for train.
    :return:
    """
    model.compile(loss="mse", optimizer="Adam", metrics=['mae'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min')  # ReduceLROnPlateau
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_data=(X_valid, y_valid),
        callbacks=[reduce_lr])

    model.save('model/gru.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/gru_loss.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    time_step, seq_len = 90, 9
    path_dict = {"train": "data/train.csv", "valid": "data/valid.csv", "test": "data/test.csv"}
    config = {"batch": 256, "epochs": 600}
    # read data
    X_train, y_train, _, X_valid, y_valid, _, X_test, y_test, _, _, _ = process_data(path_dict, time_step, seq_len)
    # build model
    units = [X_train.shape[1], X_train.shape[2], 32, 32, 1]
    gru = get_gru(units)
    # train model
    train_model(gru, X_train, y_train, X_valid, y_valid, config)
