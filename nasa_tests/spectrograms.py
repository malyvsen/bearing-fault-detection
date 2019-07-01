import os
import skimage
import numpy as np
from tqdm import tqdm
import raw_data


def spectrogram(test):
    result = [[] for i in range(test.num_channels)]
    for file in tqdm(test):
        measurements = np.array([x for x in file])
        for channel in range(test.num_channels):
            result[channel].append(np.abs(np.fft.rfft(measurements[:, channel])))
    return np.interp(result, (np.min(result), np.max(result)), (0, 1))


def save_spectrograms(tests=raw_data.tests, out_dir='spectrograms'):
    os.makedirs(out_dir, exist_ok=True)
    for test in tests:
        test_spectrogram = spectrogram(test)
        for channel in range(test.num_channels):
            spectrogram_image = test_spectrogram[channel]
            target_shape = tuple(x // 4 for x in spectrogram_image.shape)
            spectrogram_image = skimage.transform.resize(spectrogram_image, target_shape, anti_aliasing=True)
            spectrogram_image = quantile_filter(spectrogram_image)
            out_filepath = os.path.join(out_dir, f'test{test.number}_channel{channel}.png')
            skimage.io.imsave(out_filepath, spectrogram_image)


def quantile_filter(image):
    quantile_values = []
    for quantile in np.linspace(0, 1, 256):
        quantile_values.append(np.quantile(image, quantile))
    quantile_values = np.array(quantile_values)
    result = np.zeros_like(image)
    for i, row in tqdm(enumerate(image)):
        for j, pixel in enumerate(row):
            result[i, j] = np.argmin(np.abs(quantile_values - pixel))
    return result.astype(np.int32)


if __name__ == '__main__':
    save_spectrograms()
