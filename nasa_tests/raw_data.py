import os
import csv
import itertools
from datetime import datetime


class Test:
    def __init__(self, number, dir, faulty_bearings, frequency=20000):
        self.number = number
        self.path = os.path.join('data/IMS', dir)
        self.filenames = sorted(os.listdir(self.path))
        self.files = [File(self.path, filename) for filename in self.filenames]
        self.start = self.files[0].datetime
        self.end = self.files[-1].datetime
        self.duration = (self.end - self.start).total_seconds()
        self.faulty_bearings = faulty_bearings
        self.frequency = frequency

    def __iter__(self):
        return iter(self.files)


class File:
    def __init__(self, dir, name):
        self.name = name
        self.path = os.path.join(dir, name)
        self.datetime = datetime(*[int(x) for x in name.split('.')])

    def __iter__(self):
        with open(self.path, 'r') as file:
            for row in csv.reader(file, delimiter='\t'):
                yield [float(x) for x in itertools.islice(row, 0, None, 2 if len(row) == 8 else 1)]


tests = [
    Test(1, dir='1st_test', faulty_bearings=(2, 3)),
    Test(2, dir='2nd_test', faulty_bearings=(0,)),
    Test(3, dir='4th_test/txt', faulty_bearings=(2,))
]
