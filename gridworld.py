import numpy as np
from matplotlib import pyplot as plt


class GridWorld:

    actions = [
        lambda pos: pos,                 # Stay in position
        lambda pos: (pos[0], pos[1]+1),  # Move up
        lambda pos: (pos[0]+1, pos[1]),  # Move right
        lambda pos: (pos[0], pos[1]-1),  # Move down
        lambda pos: (pos[0]-1, pos[1])   # Move left
    ]

    action_strings = ['s', 'u', 'r', 'd', 'l']

    num_actions = 5

    def __init__(self):
        self.grid = -1.0 * np.ones((10, 5))
        self.grid[9, 3] = 100.0
        x = np.random.randint(0, 9)
        y = np.random.randint(0, 4)
        self.pos = (x, y)
        self.q_table = None

    def in_bounds(self, pos):
        if pos[0] < 0 or pos[0] > 9:
            return False
        if pos[1] < 0 or pos[1] > 4:
            return False
        return True

    def act(self, num_action):
        pos = GridWorld.actions[num_action](self.pos)
        if self.in_bounds(pos):
            self.pos = pos
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


def gridworld_e_greedy():
    max_e = 0.5
    num_e = 5.0
    epsilons = np.arange(0, max_e + max_e / num_e, max_e / num_e)
    e_greedy_results = np.zeros((len(epsilons), GridWorld.num_actions+1), float)
    seeds = range(1000)

    for e, epsilon in enumerate(epsilons):
        print('----Epsilon={}----'.format(epsilon))
        for seed in seeds:
            np.random.seed(seed)
            gridworld = GridWorld()
            action_num = gridworld.e_greedy(20, epsilon, 2.0)
            # e_greedy_results[e, action_num] += 1
            reward = 0
            gw = GridWorld()
            for i in range(20):
                reward += gw.act(action_num)
            e_greedy_results[e, 0] += reward
            e_greedy_results[e, action_num+1] += 1

    e_greedy_results[:, 0] /= float(len(seeds))
    print(epsilons)
    print(e_greedy_results)


def q_learning(world, episodes=20, val_initial=100.0, alpha=0.5, gamma=0.9):
    global q_table
    if q_table is None:
        q_table = val_initial * np.ones((*world.grid.shape, world.num_actions))
    for _ in range(episodes):
        best_action = np.argmax(q_table[world.pos])
        # print(GridWorld.action_strings[best_action])
        prev_value = (*world.pos, best_action)
        reward = world.act(best_action)
        q_table[prev_value] *= (1.0-alpha)
        q_table[prev_value] += alpha*(reward + gamma*np.max(q_table[world.pos]))


def evaluate_q_table():
    global q_table
    gw = GridWorld()
    total_reward = 0
    failure = 0.0
    for _ in range(20):
        total_reward += gw.act(np.argmax(q_table[gw.pos[0], gw.pos[1], :]))
    if not gw.pos == (9, 3):
        failure = 1.0
    return total_reward, failure


def gridworld_q_learning():
    np.random.seed(1)
    training_runs = 300
    fails = np.zeros((training_runs,))
    for n in range(training_runs):
        gw = GridWorld()
        q_learning(gw)
        for _ in range(100):
            score, fail = evaluate_q_table()
            fails[n] += fail

    plt.subplot(2, 1, 1)
    plt.plot(range(training_runs), fails)
    plt.title('Training Runs until Consistent Success')
    plt.xlabel('Training Run')
    plt.ylabel('test runs failed/100')
    scores = []
    for _ in range(10000):
        score, fails = evaluate_q_table()
        scores.append(score)
    plt.subplot(2, 1, 2)
    plt.title('Histogram of scores of final Q table')
    plt.ylabel('Number of trials')
    plt.xlabel('Cumulative reward')
    plt.hist(scores, edgecolor='black', linewidth=1.0)
    plt.tight_layout()
    plt.show()
    print_q_table()
    # print_q_table()


def print_q_table():
    global q_table
    X, Y, A = q_table.shape
    q_action = np.empty((X, Y), str)
    q_reward = np.empty((X, Y), float)
    for x in range(X):
        for y in range(Y):
            q_reward[x, y] = np.max(q_table[x, y])
            action = np.argmax(q_table[x, y])
            if action == 0:
                q_action[x, y] = 's'
            elif action == 1:
                q_action[x, y] = 'u'
            elif action == 2:
                q_action[x, y] = 'r'
            elif action == 3:
                q_action[x, y] = 'd'
            elif action == 4:
                q_action[x, y] = 'l'
            else:
                print("WAT")
                exit()
    print(q_action)
    # np.set_printoptions(precision=1, suppress=True)
    # print(q_reward.transpose())


if __name__ == "__main__":
    q_table = None
    # gridworld_e_greedy()
    gridworld_q_learning()
