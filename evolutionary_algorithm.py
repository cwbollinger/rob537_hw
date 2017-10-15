import numpy as np
from matplotlib import pyplot as plt
from traveling_salesman import TravelingSalesman, city_swap
import cProfile
import pstats


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
    return children


def evolutionary_algorithm(cities, population_size, iterations, seed=2):
    np.random.seed(seed)
    fitness = get_fitness_function(cities)
    population = np.empty((0, cities.num_cities), dtype=np.dtype(int))
    for i in range(population_size):
        path = np.asarray([range(cities.num_cities)])
        np.random.shuffle(path[0, :])
        population = np.concatenate((population, path), axis=0)

    for i in range(iterations):
        # generate
        parents = np.random.randint(0, len(population), (population_size,))
        children = generate_children([population[x, :] for x in parents])
        # merge
        population = np.append(population, children, axis=0)
        # evaluate
        fitnesses = np.apply_along_axis(fitness, axis=1, arr=population)
        fitnesses.sort()
        cdf = fitnesses / fitnesses.sum()
        for k in range(len(cdf)):
            if k is not 0:
                cdf[k] += cdf[k-1] # cumulative distribution
        # downsample
        indices = [np.where(p < cdf)[0][0] for p in np.random.rand(population_size)]
        population = population[indices]
    return population


if __name__ == "__main__":
    roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/15cities.csv')
    k = 20
    iterations = 1000000
    prof = cProfile.Profile()
    prof.enable()
    population = prof.runcall(evolutionary_algorithm, roadmap, k, iterations)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats('tottime')
    # population = evolutionary_algorithm(roadmap, k, iterations)
    stats.print_stats()
    diff = np.zeros((k,k))
    for i, pop in enumerate(population[x, :] for x in range(k)):
        print(roadmap.score(pop), pop)
        for j in range(k):
            d_pop =(population[j, :] - pop)
            diff[i, j] = sum(d_pop > 0)
    print('--------')
    print(diff)
