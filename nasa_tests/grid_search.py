import os
import json
import numpy as np
from analysta.cli.model import run_single



with open('lstm_config.json', 'r') as config_file:
    config = json.loads(config_file.read())

if not os.path.exists('temp'):
    os.mkdir('temp')
config['data']['train_paths_prefix'] = '../' + config['data']['train_paths_prefix']
config['data']['val_paths_prefix'] = '../' + config['data']['val_paths_prefix']
config['data']['test_paths_prefix'] = '../' + config['data']['test_paths_prefix']

accuracies = {}
for cells in 2 ** np.arange(4, 9):
    cells = int(cells) # for serializability
    config['model']['cells'] = cells
    for look_back in 2 ** np.arange(4, 9):
        look_back = int(look_back) # for serializability
        config['preparation']['look_back'] = look_back
        with open('temp/config.json', 'w') as config_file:
            json.dump(config, config_file)
        results, _, _ = run_single('temp/config.json', results_dir='temp/results')
        accuracies[(cells, look_back)] = results['out.model.test.acc']

with open('grid_search_results.json') as results_file:
    json.dump(accuracies, results_file)
