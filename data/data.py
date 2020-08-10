"""
Processing the data

"""
import numpy as np
import pandas as pd


def spilt_data(x, y, step, seq):
    """
    spilt X\y data
    :param x:ndarray
    :param y:ndarray
    :param step:int
    :param seq:int
    :return:nadarray * 3
    """
    seq_len = step + seq
    X = []
    Y = []
    y_sec = []
    for index in range(len(x) - seq_len):
        X.append(x[index: index + step])
        Y.append(y[index + seq_len] - y[index + step - 1])
        y_sec.append(y[index + step - 1])
    X = np.array(X)
    Y = np.array(Y)
    y_sec = np.array(y_sec)

    return X, Y, y_sec


def process_data(path_dict, step, seq):
    """
    read data and make train\valid\test data.
    :param path_dict:dict, name of .csv train/test/valid file.
    :param step:int, time step
    :param seq:int, sequence length
    :return:ndarray * 8, int, int
    """

    train = pd.read_csv(path_dict["train"], encoding='gbk').reset_index(drop=True)
    test = pd.read_csv(path_dict["test"], encoding='gbk').reset_index(drop=True)
    valid = pd.read_csv(path_dict["valid"], encoding='gbk').reset_index(drop=True)

    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    x_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    x_valid = valid.iloc[:, 1:]
    y_valid = valid.iloc[:, 0]

    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    x_valid = (x_valid - x_train_mean) / x_train_std

    X_train, Y_train, y = spilt_data(x_train, y_train, step, seq)
    X_valid, Y_valid, y_v = spilt_data(x_valid, y_valid, step, seq)
    X_test, Y_test, y_t = spilt_data(x_test, y_test, step, seq)

    y_train_mean = np.mean(Y_train)
    y_train_std = np.std(Y_train)
    Y_train = (Y_train - y_train_mean) / y_train_std
    Y_valid = (Y_valid - y_train_mean) / y_train_std
    Y_test = (Y_test - y_train_mean) / y_train_std

    return X_train, Y_train, y, X_valid, Y_valid, y_v, X_test, Y_test, y_t, y_train_mean, y_train_std










