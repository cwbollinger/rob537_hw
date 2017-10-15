"""
Trainer for neural networks.
Expects training set to be provided as a list where each sample is a tuple of tuples:
( (x1, x2, x3), (t1, t2) )
"""

import csv
import matplotlib.pyplot as plt
from neuron import *


class NetworkTrainer:

    @staticmethod
    def mse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return ((a-b) ** 2).mean(axis=None)

    def __init__(self, training_set, network):
        self.training_set = training_set
        self.network = network

    @staticmethod
    def classification_accuracy(confmat):
        total = np.sum(confmat)
        correct = np.sum(confmat * np.eye(2))
        return correct / total

    def train(self, iterations, eta, alpha):
        results = []
        validation_results = []
        sample_order = np.arange(len(self.training_set))
        np.random.shuffle(sample_order)
        div = int(len(sample_order) * 0.8)
        training_partition = sample_order[0:div-1]
        validation_partition = sample_order[div:-1]
        for i in range(iterations):
            np.random.shuffle(training_partition)
            # self.network.display_weights()
            result = np.zeros([2,2])
            for idx in training_partition:
                (x, t) = self.training_set[idx]
                y = self.network.evaluate(x)
                e = np.asarray(t) - np.asarray(y)
                self.network.backprop(e, eta)
                result += self.classify(t, y)
            results.append(self.classification_accuracy(result))
            validation_result = np.zeros([2,2])
            for idx in validation_partition:
                (x, t) = self.training_set[idx]
                y = self.network.evaluate(x)
                validation_result += self.classify(t, y)
            validation_results.append(self.classification_accuracy(validation_result))
            print('Progress: {:.2f}% complete'.format(100.0 * float(i) / float(iterations)))
        return [results, validation_results]

    @staticmethod
    def classify(t, y):
        result = np.zeros([2,2])
        if t[0] < t[1]:
            if y[0] < y[1]:
                result[0][0] += 1
            else:
                result[1][0] += 1
        else:
            if y[0] < y[1]:
                result[0][1] += 1
            else:
                result[1][1] += 1
        return result


def parse_dataset(filename):
    training_set = []
    with open(filename) as f:
        data_reader = csv.reader(f, delimiter=',')
        for row in data_reader:
            datapoint = (
                (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])),
                (float(row[5]), float(row[6]))
            )
            training_set.append(datapoint)
    return training_set


def classify_test(test_set, network):
    results = np.zeros([2,2])
    mse = 0
    for (x, t) in test_set:
        y = network.evaluate(x)
        mse += NetworkTrainer.mse(t, y)
        if t[0] < t[1]:
            if y[0] < y[1]:
                results[0][0] += 1
            else:
                results[1][0] += 1
        else:
            if y[0] < y[1]:
                results[0][1] += 1
            else:
                results[1][1] += 1
    mse /= len(test_set)
    return [results, mse]


def generate_comparison_graph(training_set, random_seeds, hidden_node_counts, iterations, etas, alphas):
    trainers = []
    classification_accuracies = []
    validation_classification_accuracies = []
    for i in range(3):
        acc = []
        valid_acc = []
        for j, seed in enumerate(random_seeds):
            np.random.seed(seed)
            print('----{}----'.format((i+1)*(j+1)))
            trainer = NetworkTrainer(training_set, SingleHiddenFFNN(5, hidden_node_counts[i], 2))
            [classification_accuracy, validation_classification_accuracy] = trainer.train(iterations[i], etas[i], alphas[i])
            trainers.append(trainer)
            acc.append(classification_accuracy)
            valid_acc.append(validation_classification_accuracy)
        classification_accuracies.append((np.mean(acc, axis=0), np.std(acc, axis=0)))
        validation_classification_accuracies.append((np.mean(valid_acc, axis=0), np.std(valid_acc, axis=0)))

    lines = []
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    for i in range(3):
        x = range(len(classification_accuracies[i][0]))
        y = classification_accuracies[i][0]
        yerr = classification_accuracies[i][1]
        lines.append(axes[i].errorbar(x, y, yerr, marker='.', markevery=1000, label='Training Set Accuracy'))
        x = range(len(validation_classification_accuracies[i][0]))
        y = validation_classification_accuracies[i][0]
        yerr = validation_classification_accuracies[i][1]
        lines.append(axes[i].errorbar(x, y, yerr, marker='.', markevery=1000, label='Validation Set Accuracy'))
        # axes[i].title('Hidden node count {}'.format(hidden_node_counts[i]))
        axes[1].set_xlabel('Training Epoch')
        axes[0].set_ylabel('Classification Accuracy')

    # plt.subplots_adjust(left=0.4, right=0.5)
    plt.suptitle('Hidden node count 5, 10, 15')
    plt.figlegend((lines[0][0], lines[1][0]), ('Training Set Accuracy', 'Validation Set Accuracy'), 'upper right')
    plt.show()
    return trainers


def print_performance_confmat(trainer):
    testing_set_1 = parse_dataset('C:/Users/Chris/Documents/OSU/ROB537/homework/hw1_data/test1.csv')
    testing_set_2 = parse_dataset('C:/Users/Chris/Documents/OSU/ROB537/homework/hw1_data/test2.csv')
    testing_set_3 = parse_dataset('C:/Users/Chris/Documents/OSU/ROB537/homework/hw1_data/test3.csv')
    results1, mse1 = classify_test(testing_set_1, trainer.network)
    print(results1)
    results2, mse2 = classify_test(testing_set_2, trainer.network)
    print(results2)
    results3, mse3 = classify_test(testing_set_3, trainer.network)
    print(results3)
    return results1, results2, results3


if __name__ == "__main__":
    training_set_1 = parse_dataset('C:/Users/Chris/Documents/OSU/ROB537/homework/hw1_data/train1.csv')
    training_set_2 = parse_dataset('C:/Users/Chris/Documents/OSU/ROB537/homework/hw1_data/train2.csv')
    np.random.seed(2) # 2, 22, 42
    trainers = generate_comparison_graph(training_set_2, (2, 22, 42), (10, 10, 10), (1000, 1000, 1000), (0.01, 0.09, 0.19), (0.0, 0.0, 0.0))
    for i, trainer in enumerate(trainers):
        print('--{}--'.format(i))
        results = print_performance_confmat(trainer)
