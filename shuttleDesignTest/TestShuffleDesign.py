import random

import numpy as np
import tensorflow as tf

from keras.backend import concatenate
from matplotlib import pyplot as plt

from keras.layers import Input
from keras.optimizers import SGD, Adam
import pandas as pd
from keras import layers
from CTRNNLIB.Recurrent import ShuttleCTRNN


windowSize = 1


def read_dataset(dataset_path="../../Data/denoising_signal_100.csv", window_size=20,percentage=.95):
    df = pd.read_csv(filepath_or_buffer=dataset_path)
    split = int(len(df)* percentage)
    training = df.iloc[:split, :]
    testing = df.iloc[split:, :]
    windowSize = window_size
    traininX = training.iloc[:, 0:windowSize]
    x_train = np.array(training.iloc[:, 0:windowSize])
    y_train = np.array(training.iloc[:, windowSize:windowSize + 1])

    x_test = np.array(testing.iloc[:, 0:windowSize])
    y_test = np.array(testing.iloc[:, windowSize:windowSize + 1])
    print("***************")
    print("data has been loaded successfully")
    print((x_train.shape))
    print("***************")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test








def train_deniosing_problem_rnn_window_size_not_stateful(factor=1, window_size=1, shift=0):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/1window_size_32generators.csv", window_size=window_size)

    x_train = factor * x_train+shift
    y_train = factor * y_train+shift
    x_test = factor * x_test
    y_test = factor * y_test
    epochs_ = 1
    # Any RNN layer in Keras expects a 3D shape (batch_size, timesteps, features). This means you have timeseries data.
    batch_size_ = 32
    input_layer = Input((window_size, 1))

    arg_vb = 10
    arg_input_gain = 10
    arg_output_gain = 10 ** 0
    cell1 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell2 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell3 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell4 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)
    cell5 = ShuttleCTRNN(1, vb=arg_vb, bias_initializer=tf.initializers.Zeros(), return_sequences=True, stateful=False,
                         input_gain=arg_input_gain*10, output_gain=arg_output_gain,recurrent_initializer=tf.keras.initializers.glorot_normal(seed=0))(input_layer)


    rnn_1 = concatenate([cell1, cell2, cell3, cell4, cell5])


    hidden_1 = layers.TimeDistributed(layers.Dense(5, activation='sigmoid'))(rnn_1)
    #hidden_1 = layers.Dense(5, activation='sigmoid')(rnn_1)
    output_ = layers.Dense(1, activation='linear', )(hidden_1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    model.fit(x_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/shuttle_model_weights_window_size1.h5')
    model.summary()
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)





def check_output_of_one_cell(windowSize, factor, shift):
    x_train, y_train, x_test, y_test = read_dataset(
        dataset_path="./NewTrainingData/testing_1window_size.csv", window_size=1,percentage=.5)
    batch_size_=1
    window_size=1
    input_layer = Input(batch_shape=(batch_size_, window_size, 1),
                        name="input")
    #input_layer = Input((None, 1))

    sin_signal=5*np.append(y_train,y_test)
    y_train
    y_train=5*y_train
    #plt.plot(y_train, '-')
    #plt.show()
    rnn_1 = ShuttleCTRNN(1, kernel_initializer=tf.keras.initializers.Ones(),
                         recurrent_initializer=tf.keras.initializers.Ones(),
                         vb=10,
                         bias_initializer=tf.initializers.Zeros(),return_sequences=True, stateful=False, input_gain=1,output_gain=1)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=rnn_1)  # output list [output_,...]

    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  # etrics=['mse', 'mae', 'mape']
                  metrics=['mae'],

                  )
    # train the network
    epochs_=0
    model.fit(y_train, y_train, epochs=epochs_, batch_size=batch_size_)
    model.save_weights('../../models/test.h5')
    model.summary()
    layer1 = model.layers[1].get_weights()
    layer1[1]=layer1[1]*20
    layer1[0]=layer1[0]

    model.layers[1].set_weights(layer1)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print("*" * 100)
    print("-" * 300)
    output=model.predict(y_train)
    plt.plot(output, 'k')
    plt.show()

    print(output)
if __name__ == '__main__':
    print('hi main')
    train_deniosing_problem_rnn_window_size_not_stateful(window_size=1, factor=1, shift=0)
