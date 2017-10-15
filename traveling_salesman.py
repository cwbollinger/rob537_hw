import csv
import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings('error')


class TravelingSalesman:
    def __init__(self, filename):
        with open(filename) as f:
            reader = csv.reader(f)
            num_cities = int(next(reader)[0])
            self.num_cities = num_cities
            self.cities_grid = np.zeros((num_cities, num_cities))
            cities = np.zeros((num_cities, 2))
            for i in range(num_cities):
                x, y = next(reader)
                cities[i,:] = [float(x), float(y)]
            for i in range(num_cities):
                for j in range(num_cities):
                    self.cities_grid[i,j] = np.linalg.norm(cities[j, :] - cities[i, :])

    def score(self, path):
        return np.sum(self.cities_grid[[path[:-1], path[1:]]])


def city_swap(path):
    num_cities = len(path)
    new_path = np.copy(path)
    a = np.random.randint(num_cities)
    b = np.random.randint(num_cities)
    while b == a: # this might not matter much
        b = np.random.randint(num_cities)
    new_path[a], new_path[b] = new_path[b], new_path[a]
    return new_path


if __name__ == "__main__":
    roadmap = TravelingSalesman('C:/Users/Chris/Documents/OSU/ROB537/homework/hw2_data/3cities.csv')
    print(roadmap.score([1, 0, 2]))

