import json
from model import model


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
