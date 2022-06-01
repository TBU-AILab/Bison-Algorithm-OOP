# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016
@author: hossam
"""
import math
import numpy
import random
import time
import testing
import benchmark


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = {'best': [], 'median': [], 'worst': [], 'evaluation': []}
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0


def correct_bounds(bat, dim, lb, ub):
    for d in range(dim):
        if bat[d] > ub:
            bat[d] = random.uniform(lb, ub)
        if bat[d] < lb:
            bat[d] = random.uniform(lb, ub)
    return bat


def BAT(number_of_runs, problem_definition, test_flags):
    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound = problem_definition['up_bound']
    objf = problem_definition['function']

    test_error_values = test_flags['error_values']
    test_statistics = test_flags['statistics']
    func_num = problem_definition['func_num']
    filename = problem_definition['filename']

    # BAT parameters

    n = 50;  # Population size
    # lb=-50
    # ub=50

    statistics = numpy.zeros(number_of_runs)
    if test_flags['complexity_computation']:
        max_evaluation = 200000
    else:
        max_evaluation = benchmark.get_max_fes(dimension, objf, problem_definition['self_adaptive'])
    Max_iteration = round((max_evaluation - n) / n)
    all_errors = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))
    all_diversities = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))

    # initialize solution for the final results
    s = solution()
    print("BAT is optimizing %s" % func_num)

    result = numpy.zeros(dimension)
    result_score = float("inf")

    for runs in range(number_of_runs):
        N_gen = Max_iteration  # Number of generations

        # EvoloPy Original Settings
        A = 0.5  # Loudness  (constant or decreasing)
        r = 0.5  # Pulse rate (constant or decreasing)
        Qmin = 0  # Frequency minimum
        Qmax = 2  # Frequency maximum

        dynamic_parameters = False
        alpha = 1
        gama = 1
        r0 = 0.1

        d = dimension  # Number of dimensions

        # Initializing arrays
        Q = numpy.zeros(n)  # Frequency
        v = numpy.zeros((n, d))  # Velocities
        Convergence_curve = []

        # Initialize the population/solutions
        Sol = numpy.random.rand(n, d) * (up_bound - low_bound) + low_bound
        S = numpy.zeros((n, d))
        S = numpy.copy(Sol)
        Fitness = numpy.zeros(n)

        save_errors_at = benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive'])
        evaluations = 0
        convergence_errors = []
        evaluation_curve = numpy.zeros(Max_iteration)

        # Evaluate initial random solutions
        for i in range(0, n):
            Fitness[i] = objf(Sol[i, :], dimension, func_num)
            evaluations += 1

        # Find the initial best solution
        fmin = min(Fitness)
        I = numpy.argmin(Fitness)  # corrected 14/04/2020 ! Run all experiments again!
        best = Sol[I, :]
        record_result = 0

        # Main loop
        for t in range(0, N_gen):

            # Loop over all bats(solutions)
            for i in range(0, n):
                Q[i] = Qmin + (Qmin - Qmax) * random.random()
                v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
                S[i, :] = Sol[i, :] + v[i, :]

                # Check boundaries
                # Sol=correct_bounds(Sol,d,low_bound,up_bound)
                Sol = numpy.clip(Sol, low_bound, up_bound)

                # Pulse rate
                if random.random() > r:
                    S[i, :] = best + 0.001 * numpy.random.randn(d)

                #S[i, :] = correct_bounds(S[i, :], d, low_bound, up_bound)
                S[i,:]=numpy.clip(S[i,:], low_bound, up_bound)

                # Evaluate new solutions
                Fnew = objf(S[i, :], dimension, func_num)
                evaluations += 1

                # Update if the solution improves
                if ((Fnew <= Fitness[i]) and (random.random() < A)):
                    Sol[i, :] = numpy.copy(S[i, :])
                    Fitness[i] = Fnew
                    # Following two lines are not in the EvoloPy library.
                    if dynamic_parameters:
                        A = alpha * A
                        r = r0 * (1 - numpy.exp(-gama * t))

                # Update the current best solution
                if Fnew <= fmin:
                    best = S[i, :]
                    fmin = Fnew

            if len(save_errors_at)>0 and test_error_values and evaluations >= save_errors_at[0]:
                convergence_errors.append(fmin - benchmark.known_optimum_value(func_num, objf))
                save_errors_at.pop(0)

                if test_flags['diversity']:
                    record_result += 1
                    all_diversities[runs][record_result-1]=testing.diversity_computation(Sol, n, dimension)
            if test_flags['movement_in_2d'] and t < 50:
                testing.plot_contour(filename, Sol, low_bound=low_bound, up_bound=up_bound, iteration=t,
                                     algorithm_name="BAT")

        if test_error_values:
            all_errors[runs] = numpy.array(convergence_errors)

        print(['BAT ' + str(runs) + ': [' + str(fmin) + '] Evaluations: ' + str(
            evaluations) + ' Iterations: ' + str(Max_iteration)])

        if test_statistics:
            statistics[runs] = fmin

        if result_score > fmin:
            result_score = fmin
            result = best

    if test_error_values:
        filenam = filename + '/bat_' + str(func_num) + '_' + str(dimension) + '.csv'
        testing.save_errors_to_file(all_errors, filenam)
    if test_statistics:
        statistics = testing.evaluate_all_statistics(statistics)
    if test_flags['diversity']:
        filenam = filename + '/BAT_diversity_' + str(func_num) + '_' + str(dimension) +  '.csv'
        testing.save_errors_to_file(all_diversities, filenam)


    return result_score, result
