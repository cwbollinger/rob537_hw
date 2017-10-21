import os
import csv
import cProfile
import pstats

import numpy as np

from traveling_salesman import TravelingSalesman, city_swap


def has_run(path, a, b, c):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(a, b, c)
    return os.path.exists(os.path.join(path, filename))


def save_run(paths, scores, path, a, b, c):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(a, b, c)
    with open(os.path.join(path, filename), 'w', newline='') as f:
        run_saver = csv.writer(f)
        for i, score in enumerate(scores):
            run_saver.writerow([score, *(paths[i])])


def load_run(path, a, b, c):
    filename = '{}_{:.0f}_{:.3f}.csv'.format(a, b, c)
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


# Fitness function G(x) = 1/x where x is distance traveled
def get_fitness_function(cities):
    def fitness_function(path):
        return 1.0 / cities.score(path)
    return fitness_function


def generate_children(parents):
    count = len(parents[0])
    children = np.empty([0, count], dtype=np.dtype(int))
    for parent in parents:
        child = city_swap(parent)
        child = child.reshape([1, count])
        children = np.concatenate([children, child], axis=0)
        child = city_swap(parent)
        child = city_swap(child)
        child = child.reshape([1, count])
        children = np.concatenate([children, child], axis=0)
    return children


def tourney(players, fitness):
    fitnesses = np.apply_along_axis(fitness, axis=1, arr=players)
    idx = np.where(fitnesses == fitnesses.max())
    return idx[0][0]


def tournament_select(population, num_select, fitness):
    tournament_size = int(round(len(population) / num_select))
    np.random.shuffle(population)
    idxs = [tourney(population[i:i + tournament_size, :], fitness)+i for i in range(0, len(population), tournament_size)]
    return population[idxs, :]


def proportional_fitness_select(population, num_select, fitness):
    fitnesses = np.apply_along_axis(fitness, axis=1, arr=population)
    fitnesses.sort()
    cdf = fitnesses / fitnesses.sum()
    for k in range(len(cdf)):
        if k is not 0:
            cdf[k] += cdf[k-1] # cumulative distribution
    # downsample
    indices = [np.where(p < cdf)[0][0] for p in np.random.rand(num_select)]
    return population[indices]


def evolutionary_algorithm(cities, population_size, iterations, seed=2):
    np.random.seed(seed)
    fitness = get_fitness_function(cities)
    population = np.empty((0, cities.num_cities), dtype=np.dtype(int))
    for i in range(population_size):
        path = np.asarray([range(cities.num_cities)])
        np.random.shuffle(path[0, :])
        population = np.concatenate((population, path), axis=0)

    for i in range(iterations):
        if (i % 100) == 0:
            print(i)
        # generate

        children = generate_children([population[p, :] for p in range(population_size)])
        # merge
        population = np.append(population, children, axis=0)
        # select
        # population = proportional_fitness_select(population, population_size, fitness)
        population = tournament_select(population, population_size, fitness)
    paths = [population[i, :] for i in range(population_size)]
    scores = [cities.score(path) for path in paths]
    return paths, scores


def get_ten(problem_map, pop_size, iterations):
    seeds = (2, 4, 6, 8, 10, 12, 14, 18, 20, 22)
    file_location = '/home/chris/rob537_hw/data/hw2_runs/evolutionary/{}'.format(problem_map.num_cities)
    paths = []
    scores = []
    for num, seed in enumerate(seeds):
        print('Run #{}'.format(num))
        file_params = (file_location, pop_size, iterations, seed)
        if has_run(*file_params):
            path, score = load_run(*file_params)
        else:
            np.random.seed(seed)
            path, score = evolutionary_algorithm(problem_map, pop_size, iterations, seed)
            save_run(path, score, *file_params)
        print(path)
        print(score)
        paths.append(path)
        scores.append(score)
    return paths, scores


if __name__ == "__main__":
    # roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/15cities.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities.csv')
    roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/25cities_A.csv')
    # roadmap = TravelingSalesman('/home/chris/rob537_hw/data/hw2_data/100cities.csv')
    k = 50
    iterations = 3000
    prof = cProfile.Profile()
    prof.enable()
    pops, scores = prof.runcall(get_ten, roadmap, k, iterations)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats('tottime')
    stats.print_stats()
    print([min(score) for score in scores])
    # scores = [roadmap.score(population[x, :]) for x in range(k)]
    # print(max(scores))
    '''
    diff = np.zeros((k, k))
    for i, pop in enumerate(population[x, :] for x in range(k)):
        print(roadmap.score(pop), pop)
        for j in range(k):
            d_pop =(population[j, :] - pop)
            diff[i, j] = sum(d_pop > 0)
    print('--------')
    print(diff)
    '''
