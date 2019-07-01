import os
import numpy as np
from tqdm import tqdm
import raw_data


def preprocess_data(tests=raw_data.tests, out_path='data/preprocessed', out_frequencies=(20000, 400, 200, 100), files_per_faultiness=8):
    for test in tests:
        zero_faultiness, full_faultiness = read_test(test)
        print(f'Saving preprocessed data...')
        for out_frequency in tqdm(out_frequencies):
            out_path_full = os.path.join(out_path, f'{out_frequency}_hz')
            os.makedirs(out_path_full, exist_ok=True)
            frequency_ratio = test.frequency // out_frequency
            for chunk_id, chunk in enumerate(np.array_split(zero_faultiness[::frequency_ratio], files_per_faultiness)):
                np.save(os.path.join(out_path_full, f'test_{test.number}_ok_{chunk_id}.npy'), chunk)
            for chunk_id, chunk in enumerate(np.array_split(full_faultiness[::frequency_ratio], files_per_faultiness)):
                np.save(os.path.join(out_path_full, f'test_{test.number}_fault_{chunk_id}.npy'), chunk)


def read_test(test):
    print(f'Reading data from test {test.number}')
    zero_faultiness = []
    full_faultiness = []
    for file in tqdm(test):
        seconds_remaining = (test.end - file.datetime).total_seconds()
        faultiness_value = faultiness(file.datetime, test)
        for measurement in file:
            filtered_measurement = measurement[::len(measurement) // 4]
            augmented_measurement = [seconds_remaining, faultiness_value] + filtered_measurement
            if faultiness_value == 0:
                zero_faultiness.append(augmented_measurement)
            elif faultiness_value == 1:
                full_faultiness.append(augmented_measurement)
    return np.array(zero_faultiness), np.array(full_faultiness)


# faultiness is 0 when we're sure the system is ok, 1 when we're sure it's not, otherwise in-between
def faultiness(datetime, test):
    last_zero = test.duration * 0.5
    first_one = max(test.duration * 0.9, test.duration - 3 * 24 * 60 * 60)
    return np.interp((datetime - test.start).total_seconds(), (last_zero, first_one), (0, 1))



if __name__ == '__main__':
    preprocess_data()
