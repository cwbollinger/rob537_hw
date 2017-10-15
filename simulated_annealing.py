import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from traveling_salesman import TravelingSalesman, city_swap


def simulated_annealing(problem_map, T_initial, alpha):
    paths = []
    scores = []
    T = T_initial
    path = np.asarray(range(problem_map.num_cities))
    np.random.shuffle(path)
    paths.append(path)
    curr = problem_map.score(path)
    scores.append(curr)
    # track how long it's been since a better solution was found
    steps_since_improve = 0
    best_solve = curr
    while steps_since_improve < 5000:
        new_path = city_swap(path)
        new = problem_map.score(new_path)

        if new < curr:
            path = new_path
            curr = new
        else:
            if (np.log(np.random.rand()) * T) < (curr - new):
                path = new_path
                curr = new

        scores.append(curr)
        paths.append(path)

        if curr < best_solve:
            steps_since_improve = 0
            best_solve = curr
        else:
            steps_since_improve += 1
        T *= alpha

    return paths, scores


def has_run(path, seed, t_i, dt):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(seed, t_i, dt)
    return os.path.exists(os.path.join(path, 'annealing', filename))


def save_run(path, seed, t_i, alpha, paths, scores):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(t_i, alpha, seed)
    with open(os.path.join(path, 'annealing', filename), 'w', newline='') as f:
        run_saver = csv.writer(f)
        for i, score in enumerate(scores):
            run_saver.writerow([score, *(paths[i])])


def load_run(path, seed, t_i, alpha):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(t_i, alpha, seed)
    try:
        with open(os.path.join(path, 'annealing', filename), 'r') as f:
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


def get_ten(problem_map, t_i, alpha):
    seeds = (2, 4, 6, 8, 10, 12, 14, 18, 20, 22)
    file_location = 'C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_runs'
    paths = []
    scores = []
    for seed in seeds:
        file_params = (file_location, t_i, alpha, seed)
        if has_run(*file_params):
            path, score = load_run(*file_params)
        else:
            np.random.seed(seed)
            path, score = simulated_annealing(problem_map, t_i, alpha)
        paths.append(path)
        scores.append(score)
    return paths, scores


if __name__ == "__main__":
    roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/15cities.csv')
    num_temp_vars = 20
    temp_stride = 10000000
    num_alpha_vars = 16
    alpha_stride = 0.005

    steps = [temp_stride * (x+1) for x in range(num_temp_vars)]
    alpha = [0.85 + alpha_stride * y for y in range(num_alpha_vars)]
    results = np.zeros((num_temp_vars, num_alpha_vars))
    result_errs = np.zeros((num_temp_vars, num_alpha_vars))
    for i, step in enumerate(steps):
        for j, dT in enumerate(alpha):
            print("{}/{}, {}/{}".format(i+1, num_temp_vars,j+1, num_alpha_vars))
            routes, scores = get_ten(roadmap, steps[i], alpha[j])
            best_routes = [x[-1] for x in routes]
            best_scores = [y[-1] for y in scores]
            avg = np.mean(best_scores)
            results[i,j] = avg
            std_err = np.std(best_scores) / np.sqrt(len(best_scores))
            result_errs[i,j] = std_err

    # plt.subplot(2, 1, 1)
    # plt.imshow(results, origin='lower', extent=[alpha[-1], alpha[0], steps[0], steps[-1]], aspect=(alpha_stride/temp_stride)*(num_alpha_vars/num_temp_vars))
    plt.imshow(results, origin="bottom")
    plt.title('Average Score (10 runs)')
    plt.xticks(range(len(alpha)), np.round(alpha))
    plt.xlabel('cooling factor')
    plt.yticks(range(len(steps)), steps)
    plt.ylabel('initial temperature')
    plt.colorbar()
    """
    plt.subplot(2, 1, 2)
    plt.matshow(result_errs, origin='lower')
    # plt.imshow(result_errs, origin='lower', extent=[alpha[-1], alpha[0], steps[0], steps[-1]], aspect=(alpha_stride/temp_stride))
    plt.title('Standard Error')
    plt.xlabel('cooling factor')
    plt.ylabel('initial temperature')
    # plt.yticks(steps, steps)
    plt.colorbar()
    """
    plt.show()

