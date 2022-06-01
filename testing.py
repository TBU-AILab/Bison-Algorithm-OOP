import matplotlib.pyplot as plt
import benchmark
import numpy as np
import csv


def diversity_computation(population, population_size, dimension):
    means = np.zeros(dimension, dtype=np.double)
    for d in range(dimension):
        for x in range(population_size):
            means[d] += population[x][d]
        means[d] = means[d] / population_size
    diversity = 0
    for x in range(population_size):
        for d in range(dimension):
            diversity = diversity + (population[x][d] - means[d]) * (population[x][d] - means[d])
    diversity = np.sqrt(diversity / population_size)
    return diversity


# use by: testing.plot_contour(filename, bisons, low_bound, up_bound, 0, number_of_elite_bisons, number_of_swarming_bisons)
def plot_contour(filename, positions, center=[0, 0], low_bound=-100, up_bound=100, iteration=1, number_of_elite=20,
                 number_of_swarm=40, algorithm_name="Bison", dynamic_map=False):
    fig = plt.figure()
    name_of_function = benchmark.name_of_function
    X, Y, Z = define_objective_function(name_of_function, low_bound, up_bound, iteration, dynamic_map)

    if algorithm_name == "Bison":
        for i in range(0, len(positions)):
            if i >= number_of_elite and i < number_of_swarm:
                swarming, = plt.plot(positions[i][0], positions[i][1], 'bo', fillstyle='none')
            elif i < number_of_elite:
                elites, = plt.plot(positions[i][0], positions[i][1], 'bo', fillstyle='full')
            else:  # if i > number_of_swarm:
                running, = plt.plot(positions[i][0], positions[i][1], 'ko')
        center_point, = plt.plot(center[0], center[1], 'rX')
        plt.legend([elites, swarming, running, center_point],
                   ["Elite group", "Swarming group", "Running group", "Center"], loc=2)
    else:
        for i in range(0, len(positions)):
            solutions, = plt.plot(positions[i][0], positions[i][1], 'ko')
        plt.legend([solutions],
                   [algorithm_name], loc=2)

    cf = plt.contourf(X, Y, Z, cmap=plt.cm.spring)
    plt.axis([low_bound, up_bound, low_bound, up_bound])
    plt.title('Iteration %s' % iteration)
    fig.tight_layout()
    fig.colorbar(cf)
    name = filename + algorithm_name + '_' + str(iteration) + '_cb.svg'
    fig.tight_layout()
    fig.savefig(name)
    name = str(filename) + 'png/_' + algorithm_name + '_' + str(iteration) + '.png'
    fig.savefig(name)
    plt.close(fig)

def define_objective_function(name_of_function, low_bound, up_bound, iteration=0, moving_peaks_map=False):
    x = np.linspace(low_bound, up_bound)
    y = np.linspace(low_bound, up_bound)
    X, Y = np.meshgrid(x, y)

    if name_of_function == 'De Jong 1':
        Z = X ** 2 + Y ** 2;
    if name_of_function == 'Rastrigin':
        Z = 20 + X ** 2 - 10 * np.cos(2 * np.pi * X) + Y ** 2 - 10 * np.cos(2 * np.pi * Y);
    if name_of_function == 'Schwefel':
        Z = -X * np.sin(np.sqrt(abs(X))) - Y * np.sin(np.sqrt(abs(Y)));
    if name_of_function == 'Rosenbrock':
        Z = (1. - X) ** 2 + 100. * (Y - X * X) ** 2
    if name_of_function == 'Easom':
        Z = -np.cos(X) * np.cos(Y) * np.exp(-(X - np.pi) ** 2 - (Y - np.pi) ** 2)
    if name_of_function == 'Michalewicz':
        m = 10
        Z = - np.sin(X) * np.sin((X**2)/np.pi) ** (2*m) - ( np.sin(Y) * np.sin((2 * Y**2)/np.pi) ** (2*m) )
    if name_of_function == 'De Jong 5':
        Z = 0.002
        m1 = [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32]
        m2 = [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
        for i in range(0, 25):
            Z += 1 / (i + (X - m1[i]) ** 6 + (Y - m2[i]) ** 6)
        Z = Z ** (-1)
    return [X, Y, Z]


def save_population_to_table(population, population_fitness, iteration):
    table = []
    for x in range(len(population_fitness)):
        table.append([population_fitness[x], population[x]])
    # write it
    with open('diversity/population_diversity' + str(iteration) + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]


def save_errors_to_file(errors, filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in errors]


def save_statistics(fitness, statistics):
    statistics.append(fitness[0])
    return statistics

def evaluate_all_statistics(statistics):
    solution = {}
    solution['min'] = min(statistics)
    solution['avg min'] = np.average(statistics)
    solution['std'] = np.std(statistics)
    return solution


def save_statistics_to_file(statistics, filename=""):
    all_runs = len(statistics['min'])

    result = {}
    result['best'] = min(statistics['min'])
    result['min'] = sum(statistics['min']) / all_runs
    result['max'] = sum(statistics['max']) / all_runs
    result['median'] = sum(statistics['median']) / all_runs
    result['average'] = sum(statistics['average']) / all_runs
    result['last_population_deviation'] = sum(statistics['deviation']) / all_runs

    variance = 0
    for x in range(len(statistics['min'])):
        variance += (statistics['min'][x] - result['min']) ** 2 / (len(statistics['min']) - 1)
    result['best_result_deviation'] = np.sqrt(variance)

    with open('statistics_' + str(filename) + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in result]

    print(result)
