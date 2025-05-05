import matplotlib.pyplot as plt
import numpy as np


def get_analyze_plot(results_filepath: str, savepath: str = "plot.png"):
    data = np.loadtxt(results_filepath, delimiter=",", skiprows=1)

    distances = data[:, 0]
    histogram_values = data[:, 1]

    plt.plot(distances, histogram_values, color="red", label="original")

    plt.xlabel("distance")
    plt.ylabel("prob density")
    plt.legend()
    plt.grid()
    plt.xlim(0, 20)

    plt.tight_layout()

    plt.savefig(savepath, dpi=300)
