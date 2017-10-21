import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from traveling_salesman import TravelingSalesman, city_swap


def has_run(path, seed, t_i, dt):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(seed, t_i, dt)
    return os.path.exists(os.path.join(path, filename))


def save_run(paths, scores, path, seed, t_i, alpha):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(t_i, alpha, seed)
    with open(os.path.join(path, filename), 'w', newline='') as f:
        run_saver = csv.writer(f)
        for i, score in enumerate(scores):
            run_saver.writerow([score, *(paths[i])])


def load_run(path, seed, t_i, alpha):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(t_i, alpha, seed)
    try:
        with open(os.path.join(path, filename), 'r') as f:
            run_loader = csv.reader(f)
            scores = []
            paths = []
            for row in run_loader:
                scores.append(float(row.pop(0)))
                paths.append([int(x) for x in row])
    except Exception as e:
        print(filename)
        raise e

    return paths, scores


def epsilon_greedy(problem_map, iterations, epsilon):
    paths = []
    scores = []
    path = np.asarray(range(problem_map.num_cities))
    np.random.shuffle(path)
    paths.append(path)
    curr = problem_map.score(path)
    scores.append(curr)
    # track how long it's been since a better solution was found
    for i in range(iterations):
        new_path = city_swap(path)
        new = problem_map.score(new_path)

        if new < curr:
            path = new_path
            curr = new
        else:
            if np.random.rand() < epsilon:
                path = new_path
                curr = new

        scores.append(curr)
        paths.append(path)

    return paths, scores


def has_run(path, a, b, c):
    filename = '{}_{:.3f}_{:.0f}.csv'.format(a, b, c)
    return os.path.exists(os.path.join(path, filename))


def save_run(paths, scores, path, a, b, c):
    filename = '{}_{:.3f}_{:.0f}.csv'.format(a, b, c)
    with open(os.path.join(path, filename), 'w', newline='') as f:
        run_saver = csv.writer(f)
        for i, score in enumerate(scores):
            run_saver.writerow([score, *(paths[i])])


def load_run(path, a, b, c):
    filename = '{}_{:.3f}_{:.0f}.csv'.format(a, b, c)
    try:
        with open(os.path.join(path, filename), 'r') as f:
            run_loader = csv.reader(f)
            scores = []
            paths = []
            for row in run_loader:
                scores.append(float(row.pop(0)))
                paths.append([int(x) for x in row])
    except Exception as e:
        print(filename)
        raise e

    return paths, scores


def get_ten(problem_map, num_iter, eps):
    seeds = (2, 4, 6, 8, 10, 12, 14, 18, 20, 22)
    file_location = '/home/chris/rob537_hw/data/hw2_runs/e-greedy/{}'.format(problem_map.num_cities)
    paths = []
    scores = []
    for seed in seeds:
        file_params = (file_location, seed, eps, num_iter)
        if has_run(*file_params):
            path, score = load_run(*file_params)
        else:
            np.random.seed(seed)
            path, score = epsilon_greedy(problem_map, num_iter, eps)
            save_run(path, score, *file_params)
        paths.append(path)
        scores.append(score)
    return paths, scores


if __name__ == "__main__":
    # roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities_A.csv')
    roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/100cities.csv')
    num_iter_vars = 15
    iter_stride = 1000
    num_epsilon_vars = 1
    epsilon_stride = 0.01

    steps = [1000 + iter_stride * x for x in range(num_iter_vars)]
    # epsilons = [0.01 + epsilon_stride * y for y in range(num_epsilon_vars)]
    epsilons = [0.001]
    results = np.zeros((num_iter_vars, num_epsilon_vars))
    result_errs = np.zeros((num_iter_vars, num_epsilon_vars))
    for i, iterations in enumerate(steps):
        for j, epsilon in enumerate(epsilons):
            print("{}/{}, {}/{}".format(i+1, num_iter_vars, j+1, num_epsilon_vars))
            routes, scores = get_ten(roadmap, iterations, epsilon)
            best_routes = [x[-1] for x in routes]
            best_scores = [y[-1] for y in scores]
            avg = np.mean(best_scores)
            results[i, j] = avg
            std_err = np.std(best_scores) / np.sqrt(len(best_scores))
            result_errs[i, j] = std_err

    print(np.where(results == results.min()))
    print(results[np.where(results == results.min())])
    plt.imshow(results, origin="bottom")
    plt.title('Standard Error (10 runs)')
    plt.xticks(range(len(epsilons)), np.round(epsilons, 3))
    plt.xlabel('Epsilon')
    plt.yticks(range(len(steps)), steps)
    plt.ylabel('iterations')
    plt.colorbar()
    plt.show()
