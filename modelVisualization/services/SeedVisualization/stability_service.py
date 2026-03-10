import numpy as np


def remove_stable(series, window=10, threshold=0.01):

    result = []

    for i in range(len(series)-window):

        seg = series[i:i+window]

        if np.var(seg) > threshold:

            result.append(series[i])

    return result