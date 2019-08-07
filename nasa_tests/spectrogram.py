import os
import skimage
import numpy as np
from tqdm import tqdm
import raw_data
from utils import quantile_filter



class Spectrogram:
    def __init__(self, test, data=None):
        self.test = test
        if data is not None:
            self.data = data
            return
        self.data = [[] for i in range(test.num_channels)]
        for file in tqdm(test):
            measurements = np.array([x for x in file])
            for channel in range(test.num_channels):
                self.data[channel].append(np.abs(np.fft.rfft(measurements[:, channel])))
        self.data = np.array(self.data)


    def normalized(self, range=(0, 1)):
        result_data = np.interp(self.data, (np.min(self.data), np.max(self.data)), range)
        return type(self)(self.test, result_data)


    def image(self, channel, downsampling=(4, 4)):
        result = self.data[channel]
        target_shape = np.array(result.shape) // np.array(downsampling)
        result = skimage.transform.resize(result, target_shape, anti_aliasing=True)
        result = quantile_filter(result)
        return result


    def save_data(self, dir='spectrogram/data'):
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.npy')
            np.save(path, self.data[channel])


    def save_images(self, dir='spectrogram/images'):
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.png')
            skimage.io.imsave(path, self.image(channel))



if __name__ == '__main__':
    for test in raw_data.tests:
        spectrogram = Spectrogram(test)
        spectrogram.save_data()
        spectrogram.save_images()
