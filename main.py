"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from model.model import identity_function
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    :param y_true: List/ndarray, ture data
    :param y_pred: List/ndarray, predicted data.
    :return: Double, result data for train.
    """
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def plot_results(y_true, y_preds):
    """
    Plot the true data and predicted data.
    :param y_true:  List/ndarray, ture data.
    :param y_preds: List/ndarray, predicted data.
    :return:
    """
    draw = pd.concat([pd.DataFrame(y_true), pd.DataFrame(y_preds)], axis=1)
    draw.iloc[7000:7500, 0].plot(figsize=(12, 6))
    draw.iloc[7000:7500, 1].plot(figsize=(12, 6))
    plt.legend(('real', 'predict'), fontsize='15')
    plt.title("valid", fontsize='30')  # 添加标题
    plt.show()


if __name__ == '__main__':
    gru = load_model('model/gru.h5', custom_objects={'identity_function': identity_function})

    time_step, seq_len = 90, 9
    path_dict = {"train": "data/train.csv", "valid": "data/valid.csv", "test": "data/test.csv"}

    # read data
    X_train, y_train, y_train_sec, X_valid, y_valid, y_valid_sec, X_test, y_test, y_test_sec, y_mean, y_std = \
        process_data(path_dict, time_step, seq_len)

    # run NN
    y_train_predict = gru.predict(X_train)
    y_valid_predict = gru.predict(X_valid)
    y_test_predict = gru.predict(X_test)

    y_train_predict = y_train_predict[:, -1] * y_std + y_mean + y_train_sec  # 预测出的当前时刻90s后预测值
    y_train = y_train * y_std + y_mean + y_train_sec  # 当前时刻90s后的label值

    y_valid_predict = y_valid_predict[:, -1] * y_std + y_mean + y_valid_sec
    y_valid = y_valid * y_std + y_mean + y_valid_sec

    y_test_predict = y_test_predict[:, -1] * y_std + y_mean + y_test_sec
    y_test = y_test * y_std + y_mean + y_test_sec

    # draw plt
    plot_results(y_train, y_train_predict)
    plot_results(y_valid, y_valid_predict)
    plot_results(y_test, y_test_predict)

    # print result
    print('训练集上的MAE/RMSE/MAPE/R2')
    print(mean_absolute_error(y_train, y_train_predict))  # 平均绝对误差
    print(np.sqrt(mean_squared_error(y_train, y_train_predict)))  # 均方误差
    print(MAPE(y_train, y_train_predict))
    print(r2_score(y_train, y_train_predict))
    print('验证集上的MAE/RMSE/MAPE/R2')
    print(mean_absolute_error(y_valid, y_valid_predict))
    print(np.sqrt(mean_squared_error(y_valid, y_valid_predict)))
    print(MAPE(y_valid, y_valid_predict))
    print(r2_score(y_valid, y_valid_predict))
    print('测试集上的MAE/RMSE/MAPE/R2')
    print(mean_absolute_error(y_test, y_test_predict))
    print(np.sqrt(mean_squared_error(y_test, y_test_predict)))
    print(MAPE(y_test, y_test_predict))
    print(r2_score(y_test, y_test_predict))







