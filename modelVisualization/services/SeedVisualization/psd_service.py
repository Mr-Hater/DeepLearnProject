from scipy.signal import welch
import numpy as np


def compute_psd(series):

    series = np.array(series)

    f, p = welch(series, fs=200)

    return f.tolist(), p.tolist()