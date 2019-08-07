import os
import csv
from datetime import datetime


class Test:
    def __init__(self, number, dir, num_channels, fault_channels, frequency=20000):
        self.number = number
        self.path = os.path.join('data/IMS', dir)
        self.filenames = sorted(os.listdir(self.path))
        self.files = [File(self.path, filename) for filename in self.filenames]
        self.start = self.files[0].datetime
        self.end = self.files[-1].datetime
        self.duration = (self.end - self.start).total_seconds()
        self.num_channels = num_channels
        self.fault_channels = fault_channels
        self.frequency = frequency

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)


class File:
    def __init__(self, dir, name):
        self.name = name
        self.path = os.path.join(dir, name)
        self.datetime = datetime(*[int(x) for x in name.split('.')])

    def __iter__(self):
        with open(self.path, 'r') as file:
            for row in csv.reader(file, delimiter='\t'):
                yield [float(x) for x in row]


tests = [
    Test(1, dir='1st_test', num_channels=8, fault_channels=(4, 5, 6, 7)),
    Test(2, dir='2nd_test', num_channels=4, fault_channels=(0,)),
    Test(3, dir='4th_test/txt', num_channels=4, fault_channels=(2,))
]
