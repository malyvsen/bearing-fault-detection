import json
import numpy as np
from tqdm import tqdm
from analysta.cli.model import run_single



with open('lstm_config.json', 'r') as config_file:
    config = json.loads(config_file.read())


accuracies = {}
for cells in tqdm(2 ** np.arange(4, 9)):
    cells = int(cells) # for serializability
    config['model']['cells'] = [cells]

    for look_back in tqdm(2 ** np.arange(4, 9)):
        look_back = int(look_back) # for serializability
        config['preparation']['look_back'] = look_back
        with open('temp/config.json', 'w') as config_file:
            json.dump(config, config_file)
        results, _, _ = run_single('temp_config.json', results_dir='temp_results')
        accuracies[f'{(cells, look_back)}'] = results['out.model.test.acc']

with open('grid_search_results.json', 'w') as results_file:
    json.dump(accuracies, results_file)
