import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def split_sequences(sequences, ins, out):
    sequences = np.array(sequences)
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + ins
        out_end_ix = end_ix + out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def update_weight(model, demands, past, future):
    X, y = split_sequences(demands, past, future)
    model.fit(X, y, epochs=10, verbose=0)


def get_model(init_data, past, future, threshold, use_saved=False):
        if use_saved:
            try:
                print("Loaded Model ...")
                model = tf.keras.models.load_model('../models/init.hdf5')
                return model
            except:
                pass
        else:
            print("Starting to make model ...")
            X, y = split_sequences(init_data, past, future)
            model = Sequential()
            model.add(LSTM(1024, activation='relu', input_shape=(past, threshold)))
            model.add(RepeatVector(future))
            model.add(LSTM(1024, activation='relu', return_sequences=True))
            model.add(LSTM(1024, activation='relu'))
            model.add(RepeatVector(future))
            model.add(LSTM(512, activation='relu', return_sequences=True))
            model.add(LSTM(256, activation='relu'))
            model.add(RepeatVector(future))
            model.add(LSTM(256, activation='relu', return_sequences=True))
            model.add(TimeDistributed(Dense(2*threshold)))
            model.add(TimeDistributed(Dense(threshold)))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='mse')
            print("Model is compiled, starting to train model..")
            model.fit(X, y, epochs=20, verbose=0)
            print("Model fitting complete...")
            model.save("./models/init.hdf5")
        print("Model saved to ./models dir ...")
        return model

def predict_demand(model, demands):
    demands = np.array(demands)
    demands = demands.reshape((1, demands.shape[0], demands.shape[1]))
    predicted_demand = model.predict(demands, verbose=0)
    return predicted_demand[0][0]