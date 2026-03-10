import matplotlib.pyplot as plt


def save_psd(freq,power,path):

    plt.figure()

    plt.plot(freq,power)

    plt.title("EEG PSD")

    plt.savefig(path)