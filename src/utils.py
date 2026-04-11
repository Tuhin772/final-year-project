import matplotlib.pyplot as plt
import os
import numpy as np


def calculate_mse(X, reconstructions):
    return np.mean(np.power(X - reconstructions, 2), axis=1)


def get_threshold(mse, percentile=95):
    return np.percentile(mse, percentile)


def save_plot(mse, threshold):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(mse, label="Reconstruction Error")
    plt.axhline(y=threshold, linestyle='--', label="Threshold")
    plt.legend()

    plt.savefig("outputs/plots/anomaly_plot.png")
    plt.show()