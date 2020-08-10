"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential


def identity_function(x):
    """
    activation of dense
    :param x:x
    :return:x
    """
    return x


def get_gru(units):
    """
    GRU(Gated Recurrent Unit)
    Build GRU Model.
    :param units: List(int), number of input, output and hidden units.
    :return: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], activatin='relu', input_shape=(units[0], units[0]), return_sequences=True))
    model.add(GRU(units[2], activatin='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation=identity_function))

    return model
