import numpy as np
from tqdm import tqdm



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


def pass_through(x):
    return x
