import json
import tensorflow as tf
from tensorflow import keras


with open('train_one_channel.json') as config_file:
    config = json.load(config_file)


model = keras.models.Sequential()
model.add(keras.layers.LSTM(config['model']['cells'][0], input_dim=1))
model.add(Dense(config['preparation']['out_buckets'], activation='softmax'))
model.load_weights('results_one_channel/model.h5')


num_units = int(model.layers[0].weights[0].shape[1]) / 4)
assert num_units == config['model']['cells'][0]

W_x = model.layers[0].get_weights()[0]
W_h = model.layers[0].get_weights()[1]
b = model.layers[0].get_weights()[2]

result = {}
result['W_ix'] = W_x[:, :units]
result['W_fx'] = W_x[:, units: units * 2]
result['W_gx'] = W_x[:, units * 2: units * 3]
result['W_ox'] = W_x[:, units * 3:]

result['W_ih'] = W_h[:, :units]
result['W_fh'] = W_h[:, units: units * 2]
result['W_gh'] = W_h[:, units * 2: units * 3]
result['W_oh'] = W_h[:, units * 3:]

result['b_i ']= b[:units]
result['b_f ']= b[units: units * 2]
result['b_g ']= b[units * 2: units * 3]
result['b_o ']= b[units * 3:]


with open('results_one_channel/mdoel_cut.json', 'w') as result_file:
    json.dump(result, result_file)
