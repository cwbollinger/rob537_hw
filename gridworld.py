import numpy as np
from matplotlib import pyplot as plt

actions = {
    0: lambda x: x,
    1: lambda x: (x[0], x[1]+1),
    2: lambda pos: (pos[0]+1, pos[1]),
    3: lambda x: (x[0], x[1]-1),
    4: lambda pos: (pos[0]-1, pos[1])
}


class GridWorld:

    num_actions = 5

    def __init__(self):
        self.grid = -1.0 * np.ones((10, 5))
        self.grid[9, 3] = 100.0
        x = np.random.randint(0,9)
        y = np.random.randint(0,4)
        self.pos = (x, y)

    def act(self, num_action):
        self.pos = actions[num_action](self.pos)
        return self.grid[self.pos]


    def greedy(self, episodes, val_initial=100.0):
        values = val_initial * np.ones(self.num_actions)
        for i in range(episodes):
            best = np.where(values == values.max())[0][0]
            values[best] += (1.0 / (i+1.0)) * (self.act(best) - values[best])
        return np.where(values == values.max())[0][0]

    def e_greedy(self, episodes, epsilon, val_initial=100.0):
        values = val_initial * np.ones(self.num_actions)
        for i in range(episodes):
            if np.random.rand() < epsilon:
                choice = np.random.randint(0, self.num_actions)
            else:
                choice = np.where(values == values.max())[0][0]

            values[choice] += (1.0 / (i+1.0)) * (self.act(choice) - values[choice])
        return np.where(values == values.max())[0][0]


if __name__ == "__main__":
    gridworld = GridWorld()
    epoch_count = 20
    start_vals = (5.0, 10.0, 15.0, 20.0)
    epsilons = np.arange(0, 0.01, 0.001)
    e_greedy_results = np.zeros((len(start_vals), len(epsilons), gridworld.num_actions), int)
    seeds = range(1000)

    for j, start_val in enumerate(start_vals):
        print('----Number of Epochs={}, Start Value={}----'.format(epoch_count, start_val))
        for k, epsilon in enumerate(epsilons):
            for seed in seeds:
                np.random.seed(seed)
                lever_num = gridworld.e_greedy(epoch_count, epsilon, start_val)
                e_greedy_results[j, k, lever_num] += 1
