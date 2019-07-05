import os
import json
import tensorflow as tf
from tensorflow import keras


def save_weights(dir):
    weights = get_weights(load_model(dir))
    for key in weights:
        weights[key] = weights[key].tolist()
    with open(os.path.join(dir, 'model.json'), 'w') as out_file:
        json.dump(weights, out_file, sort_keys=True, indent=4)


def get_weights(model):
    num_lstm_cells = int(model.layers[0].weights[0].shape[1] // 4)

    input_weights = model.layers[0].get_weights()[0]
    hidden_weights = model.layers[0].get_weights()[1]
    lstm_biases = model.layers[0].get_weights()[2]

    result = {}
    result['W_ix'] = input_weights[:, :num_lstm_cells]
    result['W_fx'] = input_weights[:, num_lstm_cells : num_lstm_cells * 2]
    result['W_gx'] = input_weights[:, num_lstm_cells * 2 : num_lstm_cells * 3]
    result['W_ox'] = input_weights[:, num_lstm_cells * 3:]

    result['W_ih'] = hidden_weights[:, :num_lstm_cells]
    result['W_fh'] = hidden_weights[:, num_lstm_cells : num_lstm_cells * 2]
    result['W_gh'] = hidden_weights[:, num_lstm_cells * 2 : num_lstm_cells * 3]
    result['W_oh'] = hidden_weights[:, num_lstm_cells * 3:]

    result['b_i '] = lstm_biases[:num_lstm_cells]
    result['b_f '] = lstm_biases[num_lstm_cells : num_lstm_cells * 2]
    result['b_g '] = lstm_biases[num_lstm_cells * 2 : num_lstm_cells * 3]
    result['b_o '] = lstm_biases[num_lstm_cells * 3:]

    return result


def load_model(dir):
    with open(os.path.join(dir, 'train.json')) as config_file:
        config = json.load(config_file)

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(config['model']['cells'][0], input_dim=1))
    model.add(keras.layers.Dense(config['preparation']['out_buckets'], activation='softmax'))
    model.load_weights(os.path.join(dir, 'model.h5'))
    return model
