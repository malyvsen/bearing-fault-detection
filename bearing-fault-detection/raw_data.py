import os
from test import Test



tests = [
    Test(1, dir='1st_test', num_channels=8, fault_channels=(4, 5, 6, 7)),
    Test(2, dir='2nd_test', num_channels=4, fault_channels=(0,)),
    Test(3, dir='4th_test/txt', num_channels=4, fault_channels=(2,))
]


saved_tests_dir = 'data/raw'
saved_tests = []
for root, directories, filenames in os.walk(saved_tests_dir):
    for filename in filenames:
        saved_tests.append(Test.load(os.path.join(root, filename)))

for id, test in enumerate(tests):
    matching_saved = [x for x in saved_tests if x.number == test.number and x.frequency == test.frequency]
    if len(matching_saved) > 0:
        tests[id] = matching_saved[0]


def save_tests(frequencies):
    for frequency in frequencies:
        for test in tests:
            test.downsampled(frequency=frequency).save(f'{saved_tests_dir}/{frequency}hz/test{test.number}.pickle')



if __name__ == '__main__':
    save_tests(frequencies=[100, 200, 400])
