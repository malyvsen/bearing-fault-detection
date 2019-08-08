import os
import csv
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm



class Test:
    def __init__(self, number, dir, num_channels, fault_channels, original_frequency=20000):
        self.number = number
        self.dir = dir
        self.path = os.path.join('data/IMS', dir)
        self.filenames = sorted(os.listdir(self.path))
        self.files = [File(self, filename, id) for id, filename in enumerate(self.filenames)]
        self.start = self.files[0].datetime
        self.end = self.files[-1].datetime
        self.duration = (self.end - self.start).total_seconds()
        self.num_channels = num_channels
        self.fault_channels = fault_channels
        self.original_frequency = original_frequency
        self.frequency = self.original_frequency


    def downsampled(self, frequency):
        assert frequency <= self.original_frequency
        result = type(self)(self.number, self.dir, self.num_channels, self.fault_channels, self.original_frequency)
        result.frequency = frequency
        return result


    def numpy(self, read_wrapper=tqdm, cache=True):
        if hasattr(self, 'cached_numpy'):
            return self.cached_numpy
        result = []
        for file in read_wrapper(self.files):
            result.append([x for x in file])
        maxlen = max(len(x) for x in result)
        for x in result:
            x += [np.nan] * (maxlen - len(x))
        result = np.array(result)
        if cache:
            self.cached_numpy = result
        return result


    def save(self, path=None, read_wrapper=tqdm):
        if path is None:
            path = f'data/raw/{self.frequency}hz/{self.number}.pickle'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.numpy(read_wrapper=read_wrapper, cache=True)
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file:
            return pickle.load(file)


    def __iter__(self):
        return iter(self.files)


    def __len__(self):
        return len(self.files)



class File:
    def __init__(self, test, name, id):
        self.test = test
        self.name = name
        self.path = os.path.join(self.test.path, self.name)
        self.datetime = datetime(*[int(x) for x in name.split('.')])
        self.id = id


    def __iter__(self):
        frequency_counter = 0
        if hasattr(self.test, 'cached_numpy'):
            return iter(self.test.cached_numpy[self.id])
        with open(self.path, 'r') as file:
            for row in csv.reader(file, delimiter='\t'):
                frequency_counter += self.test.frequency
                if frequency_counter >= self.test.original_frequency:
                    frequency_counter -= self.test.original_frequency
                    yield [float(x) for x in row]



tests = [
    Test(1, dir='1st_test', num_channels=8, fault_channels=(4, 5, 6, 7)),
    Test(2, dir='2nd_test', num_channels=4, fault_channels=(0,)),
    Test(3, dir='4th_test/txt', num_channels=4, fault_channels=(2,))
]
