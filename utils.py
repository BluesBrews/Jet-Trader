import numpy as np
import matplotlib.pyplot as plt

def plot_portfolio(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : i + 1])
    plt.plot(x, running_avg)
    plt.title("Running average of 100 games")
    plt.savefig(figure_file)