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
        result = type(self)(self.test, self.data)
        result.data = np.interp(result.data, (np.min(result.data), np.max(result.data)), range)
        return result


    def downsampled(self, downsampling):
        result = type(self)(self.test, np.moveaxis(self.data, [0, 1, 2], [1, 2, 0]))
        target_shape = np.array(result.data.shape)[:2] // np.array(downsampling)
        result_data = skimage.transform.resize(result.data, target_shape, anti_aliasing=True)
        result.data = np.moveaxis(result.data, [1, 2, 0], [0, 1, 2])
        return result


    def image(self, channel):
        return quantile_filter(self.data[channel])


    def save_data(self, dir='data/spectrograms'):
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.npy')
            np.save(path, self.data[channel])


    def save_images(self, dir='spectrograms'):
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.png')
            skimage.io.imsave(path, self.image(channel))



if __name__ == '__main__':
    for test in raw_data.tests:
        spectrogram = Spectrogram(test)
        spectrogram.save_data()
        spectrogram.downsampled((4, 4)).save_images()
