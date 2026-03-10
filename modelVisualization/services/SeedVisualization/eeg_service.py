import numpy as np


def extract_trials(mat):

    trials = []

    for i in range(1,16):

        key = f"trial{i}"

        if key in mat:

            trials.append(np.array(mat[key]))

    return trials


def get_channel_series(trial, channel):

    # trial shape: (62, band, time)

    return trial[channel].flatten()