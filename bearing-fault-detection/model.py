import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_model(dir):
    with open(os.path.join(dir, 'train.json')) as config_file:
        config = json.load(config_file)

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(config['model']['cells'][0], input_dim=1))
    model.add(keras.layers.Dense(config['preparation']['out_buckets'], activation='softmax'))
    model.load_weights(os.path.join(dir, 'model.h5'))
    return model


def get_weights(model):
    num_lstm_cells = int(model.layers[0].weights[0].shape[1] // 4)

    lstm_input_weights = model.layers[0].get_weights()[0]
    lstm_output_weights = model.layers[0].get_weights()[1]
    lstm_biases = model.layers[0].get_weights()[2]
    dense_weights = model.layers[1].get_weights()[0]
    dense_biases = model.layers[1].get_weights()[1]

    result = {}
    result['W_ix'] = lstm_input_weights[:, :num_lstm_cells]
    result['W_fx'] = lstm_input_weights[:, num_lstm_cells : num_lstm_cells * 2]
    result['W_gx'] = lstm_input_weights[:, num_lstm_cells * 2 : num_lstm_cells * 3]
    result['W_ox'] = lstm_input_weights[:, num_lstm_cells * 3:]

    result['W_ih'] = lstm_output_weights[:, :num_lstm_cells]
    result['W_fh'] = lstm_output_weights[:, num_lstm_cells : num_lstm_cells * 2]
    result['W_gh'] = lstm_output_weights[:, num_lstm_cells * 2 : num_lstm_cells * 3]
    result['W_oh'] = lstm_output_weights[:, num_lstm_cells * 3:]

    result['b_i '] = lstm_biases[:num_lstm_cells]
    result['b_f '] = lstm_biases[num_lstm_cells : num_lstm_cells * 2]
    result['b_g '] = lstm_biases[num_lstm_cells * 2 : num_lstm_cells * 3]
    result['b_o '] = lstm_biases[num_lstm_cells * 3:]

    result['W_yh'] = dense_weights
    result['b_y'] = dense_biases

    return result


def save_weights(weights):
    for key in weights:
        weights[key] = weights[key].tolist()
    with open(os.path.join(dir, 'model.json'), 'w') as out_file:
        json.dump(weights, out_file, sort_keys=True, indent=4)


def generate_samples(model, num_samples=16):
    result = []
    for i in range(num_samples):
        input_sample = np.sin(np.linspace(0, np.random.geometric(0.05), 32))
        result.append({'input': input_sample, 'output': model.predict(np.reshape(input_sample, (1, -1, 1)))})
    return result
