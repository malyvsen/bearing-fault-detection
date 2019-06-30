import os
import csv
import itertools
import datetime
import numpy as np
from tqdm import tqdm


original_frequency = 20000 # the NASA dataset has a probing frequency of 20kHz
original_dirs = ('1st_test', '2nd_test', '4th_test/txt')


def preprocess_data(in_path='data/IMS', out_path='data/preprocessed', out_frequencies=(20000, 400, 200, 100), files_per_faultiness=8):
    for test_id, data_dir in enumerate(original_dirs):
        print(f'Preprocessing data in {data_dir}')
        data_path = os.path.join(in_path, data_dir)
        zero_faultiness, full_faultiness = read_dir(data_path)
        for out_frequency in out_frequencies:
            out_path_full = os.path.join(out_path, f'{out_frequency}_hz')
            os.makedirs(out_path_full, exist_ok=True)
            frequency_ratio = original_frequency // out_frequency
            for chunk_id, chunk in enumerate(np.array_split(zero_faultiness[::frequency_ratio], files_per_faultiness)):
                np.save(os.path.join(out_path_full, f'test_{test_id + 1}_ok_{chunk_id}.npy'), chunk)
            for chunk_id, chunk in enumerate(np.array_split(full_faultiness[::frequency_ratio], files_per_faultiness)):
                np.save(os.path.join(out_path_full, f'test_{test_id + 1}_fault_{chunk_id}.npy'), chunk)


def read_dir(dir, line_ratio=None):
    zero_faultiness = []
    full_faultiness = []
    for data_point in iterate_dir(dir, line_ratio=line_ratio):
        faultiness = data_point[2]
        if faultiness == 0:
            zero_faultiness.append(data_point)
        elif faultiness == 1:
            full_faultiness.append(data_point)
    return np.array(zero_faultiness), np.array(full_faultiness)


def iterate_dir(dir, line_ratio=None):
    filenames = sorted(os.listdir(dir))
    fault_time = filename_datetime(filenames[-1])
    total_days = (fault_time - filename_datetime(filenames[0])).total_seconds() / 60 / 60 / 24

    for filename in tqdm(filenames):
        for data_point in iterate_file(os.path.join(dir, filename), line_ratio):
            data_point_time = filename_datetime(filename)
            seconds_remaining = (fault_time - data_point_time).total_seconds()
            days_remaining = seconds_remaining / 60 / 60 / 24
            additional_channels = [float(data_point_time.strftime('%s')), days_remaining, faultiness(days_remaining, total_days)]
            yield np.array(additional_channels + data_point)


def iterate_file(file_path, line_ratio=None):
    with open(file_path, 'r') as file:
        for row in itertools.islice(csv.reader(file, delimiter='\t'), 0, None, line_ratio):
            yield [float(x) for x in itertools.islice(row, 0, None, 2 if len(row) == 8 else 1)]


def filename_datetime(filename):
    split_name = filename.split('.')
    return datetime.datetime(*[int(x) for x in split_name])


# faultiness is 0 when we're sure the system is ok, 1 when we're sure it's not, otherwise in-between
def faultiness(days_remaining, total_running_days):
    last_zero = 0.5 * total_running_days
    first_one = min(5, total_running_days * 0.1)
    return np.interp(days_remaining, (first_one, last_zero), (1, 0))



if __name__ == '__main__':
    preprocess_data()
