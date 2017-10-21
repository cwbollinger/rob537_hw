import numpy as np
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


if __name__ == "__main__":
    bandit = NArmBandit([
        (1, 5),
        (1.5, 1),
        (2, 1),
        (2, 2),
        (1.75, 10)
    ])
    print(bandit.dumb_choose(100, 0))
    print(bandit.dumb_choose(100, 1))
    print(bandit.dumb_choose(100, 2))
    print(bandit.dumb_choose(100, 3))
    print(bandit.dumb_choose(100, 4))
    epoch_counts = (10, 100)
    start_vals = (5.0, 10.0, 15.0, 20.0)
    epsilons = np.arange(0, 0.01, 0.001)
    greedy_results = np.zeros((len(epoch_counts), len(start_vals), bandit.num_levers), int)
    e_greedy_results = np.zeros((len(epoch_counts), len(start_vals), len(epsilons), bandit.num_levers), int)
    seeds = range(1000)

    for i, epoch_count in enumerate(epoch_counts):
        for j, start_val in enumerate(start_vals):
            print('----Number of Epochs={}, Start Value={}----'.format(epoch_count, start_val))
            for seed in seeds:
                np.random.seed(seed)
                lever_num = bandit.greedy(epoch_count, start_val)
                greedy_results[i, j, lever_num] += 1

            '''
            for k, epsilon in enumerate(epsilons):
                for seed in seeds:
                    np.random.seed(seed)
                    lever_num = bandit.e_greedy(epoch_count, epsilon, start_val)
                    e_greedy_results[i, j, k, lever_num] += 1
            '''

    for i, epoch_count in enumerate(epoch_counts):
        print('--Epochs: {}--'.format(epoch_count))
        for j, start_val in enumerate(start_vals):
            plt.subplot(len(epoch_counts), len(start_vals), i*len(start_vals)+(j+1))
            plt.bar(range(len(greedy_results[i, j])), greedy_results[i, j])
            plt.title('N={}, V_i={}'.format(epoch_count, start_val))
            plt.xlabel('Lever #')
            plt.ylabel('#/{} chosen'.format(len(seeds)))

    plt.tight_layout()
    plt.show()

