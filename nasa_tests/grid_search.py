import os
import csv
import json
import numpy as np
from tqdm import tqdm



with open('lstm_config.json', 'r') as config_file:
    config = json.loads(config_file.read())


accuracies = {}
for cells in 2 ** np.arange(4, 9):
    cells = int(cells) # for serializability
    config['model']['cells'] = [cells]

    for look_back in 2 ** np.arange(4, 9):
        look_back = int(look_back) # for serializability
        config['preparation']['look_back'] = look_back
        with open('temp_config.json', 'w') as config_file:
            json.dump(config, config_file)
        os.system('analysta -vv model single -c temp_config.json')
        for filename in os.listdir('results'):
            if 'report' not in filename:
                continue
            with open('results/' + filename) as report_file:
                reader = csv.DictReader(report_file)
                for line in reader:
                    accuracies[f'{(cells, look_back)}'] = line['out.model.test.acc']

with open('grid_search_results.json', 'w') as results_file:
    json.dump(accuracies, results_file)
