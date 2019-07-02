import os
import numpy as np
from tqdm import tqdm
import raw_data


def preprocess_data(tests=raw_data.tests, out_path='data/preprocessed', out_frequencies=(20000, 400, 200, 100), num_chunks=8):
    for test in tests:
        zero_faultiness, full_faultiness = read_test(test)
        print(f'Saving preprocessed data...')
        for out_frequency in tqdm(out_frequencies):
            save_data(out_path, test, zero_faultiness, faultiness=0, frequency=out_frequency, num_chunks=num_chunks)
            save_data(out_path, test, full_faultiness, faultiness=1, frequency=out_frequency, num_chunks=num_chunks)


def read_test(test):
    print(f'Reading data from test {test.number}')
    zero_faultiness = []
    full_faultiness = []
    for file in tqdm(test):
        seconds_remaining = (test.end - file.datetime).total_seconds()
        faultiness_value = faultiness(file.datetime, test)
        for measurement in file:
            augmented_measurement = [seconds_remaining, faultiness_value] + measurement
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


def save_data(dir, test, array, faultiness, frequency, num_chunks):
    faultiness_text = 'ok' if faultiness == 0 else 'fault'
    frequency_ratio = test.frequency // frequency
    frequency_dir = os.path.join(dir, f'{frequency}hz')
    four_channel_dir = os.path.join(frequency_dir, 'four_channel')
    os.makedirs(four_channel_dir, exist_ok=True)
    one_channel_dir = os.path.join(frequency_dir, 'one_channel')
    os.makedirs(one_channel_dir, exist_ok=True)
    for chunk_id, chunk in enumerate(np.array_split(array[::frequency_ratio], num_chunks)):
        kept_indices = np.array([0, 1] + [channel * (test.num_channels // 4) + 2 for channel in range(4)])
        np.save(os.path.join(four_channel_dir, f'test{test.number}_{faultiness_text}{chunk_id}.npy'), chunk)
        for channel in range(test.num_channels):
            kept_indices = np.array([0, 1, channel + 2])
            np.save(os.path.join(one_channel_dir, f'test{test.number}_channel{channel}_{faultiness_text}{chunk_id}.npy'), chunk[:, kept_indices])


if __name__ == '__main__':
    preprocess_data()
