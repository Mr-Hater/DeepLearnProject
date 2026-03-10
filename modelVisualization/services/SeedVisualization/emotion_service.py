import scipy.io
import numpy as np
from django.conf import settings


def load_labels():

    mat = scipy.io.loadmat(settings.DATA_ROOT + "/label.mat")

    return mat["label"].flatten()


def emotion_average(trials):

    labels = load_labels()

    pos = []
    neu = []
    neg = []

    for i,t in enumerate(trials):

        if labels[i]==1:
            pos.append(t)

        elif labels[i]==0:
            neu.append(t)

        else:
            neg.append(t)

    return {

    "positive":np.mean(pos,axis=0).tolist(),

    "neutral":np.mean(neu,axis=0).tolist(),

    "negative":np.mean(neg,axis=0).tolist()

    }