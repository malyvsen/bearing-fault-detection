import os
import csv
import itertools
import datetime
import numpy as np
from tqdm import tqdm


def process_data(in_path='data/IMS', out_path='data/processed', out_frequency=400):
    line_ratio = 20000 // out_frequency # the NASA dataset has a probing frequency of 20kHz

    for id, data_dir in enumerate(('1st_test', '2nd_test', '4th_test/txt')):
        data_path = os.path.join(in_path, data_dir)
        print(f'Finding data shape in {data_dir}')
        data_shape = read_dir(data_path, out_array=None, line_ratio=line_ratio)

        data = np.zeros(shape=data_shape, dtype=np.float32)
        print(f'Processing data in {data_dir}')
        read_dir(data_path, out_array=data, line_ratio=line_ratio)
        np.save(os.path.join(out_path, f'test_{id + 1}.npy'), data)


def read_dir(dir, out_array=None, line_ratio=None):
    filenames = sorted(os.listdir(dir))
    fault_time = filename_datetime(filenames[-1])

    write_index = 0
    num_channels = 0
    for filename in tqdm(filenames):
        for data_point in iterate_file(os.path.join(dir, filename), line_ratio):
            if out_array is not None:
                data_point_time = filename_datetime(filename)
                seconds_until_fault = (fault_time - data_point_time).total_seconds()
                days_until_fault = seconds_until_fault / 60 / 60 / 24
                additional_channels = [float(data_point_time.strftime('%s')), days_until_fault, faultiness(days_until_fault)]
                out_array[write_index] = np.array(additional_channels + data_point)
            write_index += 1
            num_channels = len(data_point) + 3
    return write_index, num_channels


def iterate_file(file_path, line_ratio=None):
    with open(file_path, 'r') as file:
        for row in itertools.islice(csv.reader(file, delimiter='\t'), 0, None, line_ratio):
            yield [float(x) for x in row]


def filename_datetime(filename):
    split_name = filename.split('.')
    return datetime.datetime(*[int(x) for x in split_name])


# faultiness is an arbitrary value used to represent how bad a bearing system is doing
# it is close to 0 most of the time, but rises to 1 when a fault is near
def faultiness(days_until_fault, detectability_point=4, steepness=2):
    return 1 / (1 + np.exp((-days_until_fault - detectability_point) * steepness))



if __name__ == '__main__':
    process_data()
