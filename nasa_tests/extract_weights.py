import json
import tensorflow as tf
from tensorflow import keras


with open('train_one_channel.json') as config_file:
    config = json.load(config_file)


model = keras.models.Sequential()
model.add(keras.layers.LSTM(config['model']['cells'][0], input_dim=1))
model.add(keras.layers.Dense(config['preparation']['out_buckets'], activation='softmax'))
model.load_weights('results_one_channel/model.h5')


num_units = int(model.layers[0].weights[0].shape[1] // 4)
assert num_units == config['model']['cells'][0]

W_x = model.layers[0].get_weights()[0]
W_h = model.layers[0].get_weights()[1]
b = model.layers[0].get_weights()[2]

result = {}
result['W_ix'] = W_x[:, :num_units]
result['W_fx'] = W_x[:, num_units : num_units * 2]
result['W_gx'] = W_x[:, num_units * 2 : num_units * 3]
result['W_ox'] = W_x[:, num_units * 3:]

result['W_ih'] = W_h[:, :num_units]
result['W_fh'] = W_h[:, num_units : num_units * 2]
result['W_gh'] = W_h[:, num_units * 2 : num_units * 3]
result['W_oh'] = W_h[:, num_units * 3:]

result['b_i '] = b[:num_units]
result['b_f '] = b[num_units : num_units * 2]
result['b_g '] = b[num_units * 2 : num_units * 3]
result['b_o '] = b[num_units * 3:]


for key in result:
    result[key] = result[key].tolist()

with open('results_one_channel/model_cut.json', 'w') as result_file:
    json.dump(result, result_file, sort_keys=True, indent=4)
