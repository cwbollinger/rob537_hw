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
    while steps_since_improve < 1000:
        new_path = city_swap(path)
        new = problem_map.score(new_path)

        if new < curr:
            steps_since_improve = 0
            path = new_path
            curr = new
        else:
            steps_since_improve += 1
            if (np.log(np.random.rand()) * T) < (curr - new):
                path = new_path
                curr = new

        scores.append(curr)
        paths.append(path)

        T *= alpha

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


def get_ten(problem_map, t_i, cf):
    seeds = (2, 4, 6, 8, 10, 12, 14, 18, 20, 22)
    file_location = '/home/chris/rob537_hw/data/hw2_runs/annealing/{}_A'.format(problem_map.num_cities)
    paths = []
    scores = []
    for seed in seeds:
        file_params = (file_location, seed, cf, t_i)
        if has_run(*file_params):
            path, score = load_run(*file_params)
        else:
            np.random.seed(seed)
            path, score = simulated_annealing(problem_map, t_i, cf)
            save_run(path, score, *file_params)
        paths.append(path)
        scores.append(score)
    return paths, scores


if __name__ == "__main__":
    # roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities.csv')
    roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities_A.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/100cities.csv')
    num_temp_vars = 15
    temp_stride = 10000000
    num_alpha_vars = 13
    alpha_stride = 0.005

    temps = [temp_stride * (x+1) for x in range(num_temp_vars)]
    alphas = [0.85 + alpha_stride * y for y in range(num_alpha_vars)]
    results = np.zeros((num_temp_vars, num_alpha_vars))
    result_errs = np.zeros((num_temp_vars, num_alpha_vars))
    for i, temp in enumerate(temps):
        for j, cooling_factor in enumerate(alphas):
            print("{}/{}, {}/{}".format(i+1, num_temp_vars, j+1, num_alpha_vars))
            routes, scores = get_ten(roadmap, temp, cooling_factor)
            best_routes = [x[-1] for x in routes]
            best_scores = [y[-1] for y in scores]
            avg = np.mean(best_scores)
            results[i, j] = avg
            std_err = np.std(best_scores) / np.sqrt(len(best_scores))
            result_errs[i, j] = std_err

    print(np.where(results == results.min()))
    print(results)
    print(results[np.where(results == results.min())])
    plt.imshow(result_errs, origin="bottom")
    # plt.imshow(results, origin="bottom")
    plt.title('Average Score (10 runs)')
    plt.xticks(range(len(alphas)), np.round(alphas, 3))
    plt.xlabel('cooling factor')
    plt.yticks(range(len(temps)), temps)
    plt.ylabel('initial temperature')
    plt.colorbar()
    plt.show()
