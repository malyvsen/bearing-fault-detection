import os
import skimage
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import raw_data
import utils



class Spectrogram:
    def __init__(self, test, data=None):
        self.test = test
        if data is not None:
            self.data = data
            return
        self.data = [[] for i in range(test.num_channels)]
        for file in tqdm(test):
            measurements = file.numpy()
            for channel in range(test.num_channels):
                self.data[channel].append(np.abs(np.fft.rfft(measurements[:, channel])))
        self.data = np.array(self.data)


    def normalized(self, range=(0, 1)):
        result = type(self)(self.test, self.data)
        result.data = np.interp(result.data, (np.min(result.data), np.max(result.data)), range)
        return result


    def reshaped(self, downsampling=None, target_shape=None):
        assert sum([downsampling is not None, target_shape is not None]) == 1
        result = type(self)(self.test, np.moveaxis(self.data, [0, 1, 2], [2, 0, 1]))
        if downsampling is not None:
            target_shape = np.array(result.data.shape)[:2] // np.array(downsampling)
        result.data = skimage.transform.resize(result.data, target_shape, anti_aliasing=True)
        result.data = np.moveaxis(result.data, [2, 0, 1], [0, 1, 2])
        return result


    def image(self, channel, filter=utils.quantile_filter):
        return filter(self.data[channel])


    def save_data(self, dir='data/spectrograms'):
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.npy')
            np.save(path, self.data[channel])


    def plot(self, channel, filter=utils.pass_through, title=None, show=True, save_as=None, **kwargs):
        image_unfiltered = self.image(channel, filter=np.transpose)[::-1]
        image_filtered = filter(image_unfiltered)
        image_plot = plt.imshow(image_filtered, **kwargs)

        plt.xlabel('Time [h]')
        num_xticks = 8
        xticks_locations = np.linspace(0, image_filtered.shape[1], num_xticks)
        xticks_labels = [int(np.round(x)) for x in np.linspace(0, self.test.duration / 60 / 60, num_xticks)]
        plt.xticks(xticks_locations, xticks_labels)

        plt.ylabel('Frequency [Hz]')
        num_yticks = 4
        yticks_locations = np.linspace(0, image_filtered.shape[0], num_yticks)
        yticks_labels = [int(np.round(x)) for x in np.linspace(self.test.frequency / 2, 0, num_yticks)]
        plt.yticks(yticks_locations, yticks_labels)

        num_colorticks = 4
        colorticks_locations = np.linspace(np.min(image_filtered), np.max(image_filtered), num_colorticks)
        colorticks_labels = []
        for brightness in colorticks_locations:
            index = np.unravel_index(np.argmin(np.abs(image_filtered - brightness)), image_filtered.shape)
            amplitude = image_unfiltered[index]
            colorticks_labels.append('{:.1f}'.format(amplitude))
        colorbar = plt.colorbar(ticks=colorticks_locations)
        colorbar.ax.set_ylabel('Vibration [g]')
        colorbar.ax.set_yticklabels(colorticks_labels)

        plt.title(title if title is not None else f'Test {self.test.number}, channel {channel} @ {self.test.frequency} Hz')
        if show:
            plt.show()
        if save_as is not None:
            plt.savefig(save_as, bbox_inches='tight')


    def save_plots(self, dir=None, **kwargs):
        if dir is None:
            dir = f'spectrograms/{self.test.frequency}hz'
        os.makedirs(dir, exist_ok=True)
        for channel in range(self.test.num_channels):
            path = os.path.join(dir, f'test{self.test.number}_channel{channel}.svg')
            self.plot(channel, show=False, save_as=path, **kwargs)



if __name__ == '__main__':
    for test in raw_data.tests + raw_data.saved_tests:
        spectrogram = Spectrogram(test)
        spectrogram.save_data()
        spectrogram.reshaped(target_shape=(800, 400)).save_plots(filter=utils.quantile_filter)
