import json
import tensorflow as tf
from tensorflow import keras


with open('train_one_channel.json') as config_file:
    config = json.load(config_file)


model = keras.models.Sequential()
model.add(keras.layers.LSTM(config['model']['cells'][0], input_dim=1))
model.add(keras.layers.Dense(config['preparation']['out_buckets'], activation='softmax'))
model.load_weights('results_one_channel/model.h5')
