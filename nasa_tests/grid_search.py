import os
import csv
import json
import numpy as np
from tqdm import tqdm
from analysta.cli.model import run_single


def grid_search(cells_values=2 ** np.arange(4, 9), look_back_values=2 ** np.arange(4, 9)):
    result = {}
    with open('lstm_config.json', 'r') as config_file:
        config = json.loads(config_file.read())

    for cells in cells_values:
        cells = int(cells) # for serializability
        config['model']['cells'] = [cells]

        for look_back in look_back_values:
            look_back = int(look_back) # for serializability
            config['preparation']['look_back'] = look_back
            with open('temp_config.json', 'w') as config_file:
                json.dump(config, config_file)
            report, _, _ = run_single('temp_config.json')
            result[f'{(cells, look_back)}'] = report['out.model.test.acc']
    return result


if __name__ == '__main__':
    result = grid_search()
    with open('grid_search_result.json', 'w') as result_file:
        json.dump(result, results_file)
