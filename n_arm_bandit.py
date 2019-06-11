import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class NArmBandit:

    def __init__(self, levers):
        self.num_levers = len(levers)
        self.levers = [{'mean': lever[0], 'stddev': np.sqrt(lever[1])} for lever in levers]

    def pull_lever(self, k):
        lever = self.levers[k]
        return np.random.normal(lever['mean'], lever['stddev'])

    def dumb_choose(self, episodes, k):
        return sum([self.pull_lever(k) for i in range(episodes)])

    def greedy(self, episodes, val_initial=100.0):
        values = val_initial * np.ones(self.num_levers)
        for i in range(episodes):
            best = np.where(values == values.max())[0][0]
            values[best] += (1.0 / (i+1.0)) * (self.pull_lever(best) - values[best])
        return np.where(values == values.max())[0][0]

    def e_greedy(self, episodes, epsilon, val_initial=100.0):
        values = val_initial * np.ones(self.num_levers)
        for i in range(episodes):
            if np.random.rand() < epsilon:
                choice = np.random.randint(0, self.num_levers)
            else:
                choice = np.where(values == values.max())[0][0]

            values[choice] += (1.0 / (i+1.0)) * (self.pull_lever(choice) - values[choice])
        return np.where(values == values.max())[0][0]


def draw_greedy_plot(results):
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax2d = plt.subplots(nrows=len(epoch_counts), ncols=len(start_vals), sharex=True, sharey=True)
    for i, epoch_count in enumerate(epoch_counts):
        print('--Epochs: {}--'.format(epoch_count))
        for j, start_val in enumerate(start_vals):
            ax2d[i, j].bar(range(len(results[i, j])), results[i, j])
            ax2d[i, j].set_xticks(range(bandit.num_levers))

            if i == (len(epoch_counts) - 1):
                ax2d[i, j].set_xlabel('Lever #\nV_i={}'.format(start_val))

            if j == 0:
                ax2d[i, j].set_ylabel('N={}\n#/{} chosen'.format(epoch_count, len(seeds)))

    fig.tight_layout()
    plt.show(fig)


def draw_e_greedy_plot(results):
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax2d = plt.subplots(nrows=len(epoch_counts), ncols=len(epsilons), sharex=True, sharey=True)
    for i, epoch_count in enumerate(epoch_counts):
        print('--Epochs: {}--'.format(epoch_count))
        for j, epsilon in enumerate(epsilons):
            ax2d[i, j].bar(range(len(results[i, j])), results[i, j])
            ax2d[i, j].set_xticks(range(bandit.num_levers))

            if i == (len(epoch_counts) - 1):
                ax2d[i, j].set_xlabel('Lever #\nepsilon={:.3f}'.format(epsilon))

            if j == 0:
                ax2d[i, j].set_ylabel('N={}\n#/{} chosen'.format(epoch_count, len(seeds)))

    fig.tight_layout()
    plt.show(fig)


if __name__ == "__main__":
    bandit = NArmBandit([
        (1, 5),
        (1.5, 1),
        (2, 1),
        (2, 2),
        (1.75, 10)
    ])
    epoch_counts = (10, 100)
    start_vals = range(5)
    greedy_results = np.zeros((len(epoch_counts), len(start_vals), bandit.num_levers), int)
    seeds = range(1000)

    for i, epoch_count in enumerate(epoch_counts):
        for j, start_val in enumerate(start_vals):
            print('----Number of Epochs={}, Start Value={}----'.format(epoch_count, start_val))
            for seed in seeds:
                np.random.seed(seed)
                lever_num = bandit.greedy(epoch_count, start_val)
                greedy_results[i, j, lever_num] += 1
    # draw_greedy_plot(greedy_results)

    e_max = 0.7
    epsilons = np.arange(e_max / 5.0, e_max + e_max / 5.0, e_max / 5.0)
    e_greedy_results = np.zeros((len(epoch_counts), len(epsilons), bandit.num_levers), int)
    for i, epoch_count in enumerate(epoch_counts):
        for k, epsilon in enumerate(epsilons):
            for seed in seeds:
                np.random.seed(seed)
                lever_num = bandit.e_greedy(epoch_count, epsilon, 2.0)
                e_greedy_results[i, k, lever_num] += 1
    draw_e_greedy_plot(e_greedy_results)

