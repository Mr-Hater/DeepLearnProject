import os
import scipy.io
from django.conf import settings

def list_datasets():

    return ["ExtractedFeatures","Preprocessed_EEG"]


def list_mat_files(dataset):

    path = os.path.join(settings.DATA_ROOT, dataset)

    files = []

    for f in os.listdir(path):

        if f.endswith(".mat"):

            files.append(f)

    return files


def load_mat(dataset, filename):

    path = os.path.join(settings.DATA_ROOT, dataset, filename)

    mat = scipy.io.loadmat(path)

    return mat