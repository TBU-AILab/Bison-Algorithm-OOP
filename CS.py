# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016
within the EvoloPy optimization library
@author: Hossam Faris
-> Modified by Anezka Kazikova to fit the uniform template in 2018
"""
import math
import numpy
import random
import testing
import benchmark

func_num = 0

def get_cuckoos(nest, best, lb, ub, n, dim):
    # perform Levy flights
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.array(nest)
    beta = 3 / 2;
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
    math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta);

    s = numpy.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = numpy.random.randn(len(s)) * sigma
        v = numpy.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        alpha = 0.01 # I believe, this number should be the Alpha from the paper "Cuckoo Search via Levy Flight, Yang & Deb 2010"
        stepsize = alpha * (step * (s - best))

        s = s + stepsize * numpy.random.randn(len(s))

        tempnest[j, :] = numpy.clip(s, lb, ub)

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf, evaluations):
    # Evaluating all new solutions
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :], dim, func_num)
        evaluations += 1
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmin = min(fitness)
    K = numpy.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness, evaluations


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim):
    # Discovered or not
    tempnest = numpy.zeros((n, dim))

    K = numpy.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (nest[numpy.random.permutation(n), :] - nest[numpy.random.permutation(n), :])

    tempnest = nest + stepsize * K

    return tempnest


##########################################################################


def CS(number_of_runs, problem_definition, test_flags, params_set=1):
    global func_num
    # lb=-1
    # ub=1
    if params_set == 2:
        n=50
    else:
        n=20
    # N_IterTotal=1000
    # dim=30
    test_statistics = test_flags['statistics']
    test_error_values = test_flags['error_values']

    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound  = problem_definition['up_bound']
    objf      = problem_definition['function']
    func_num =  problem_definition['func_num']
    filename = problem_definition['filename']

    if test_flags['complexity_computation']:
        max_evaluation = 200000
    else:
        max_evaluation = benchmark.get_max_fes(dimension, objf, problem_definition['self_adaptive'])
    max_iteration = round(max_evaluation/n/2)
    average_convergence_curve = numpy.zeros((number_of_runs, max_iteration))
    all_errors = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))
    all_diversities = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))
    evaluations_curve = numpy.zeros(max_iteration)
    statistics = numpy.zeros(number_of_runs)
    best_cuckoo = [0] * dimension
    best_cuckoo_score = float("inf")

    for runs in range(number_of_runs):
        save_errors_at = benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive'])
        evaluations = 0
        # Discovery rate of alien eggs/solutions
        pa = 0.25

        nd = dimension

        #    Lb=[lb]*nd
        #    Ub=[ub]*nd

        # RInitialize nests randomly
        nest = numpy.random.rand(n, dimension) * (up_bound - low_bound) + low_bound

        new_nest = numpy.zeros((n, dimension))
        new_nest = numpy.copy(nest)

        bestnest = [0] * dimension

        fitness = numpy.zeros(n)
        fitness.fill(float("inf"))

        fmin, bestnest, nest, fitness, evaluations = get_best_nest(nest, new_nest, fitness, n, dimension, objf, evaluations)
        convergence_errors = []
        record_result = 0
        # Main loop counter
        for iter in range(0, max_iteration):
            # Generate new solutions (but keep the current best)
            new_nest = get_cuckoos(nest, bestnest, low_bound, up_bound, n, dimension)

            # Evaluate new solutions and find best
            fnew, best, nest, fitness, evaluations = get_best_nest(nest, new_nest, fitness, n, dimension, objf, evaluations)

            new_nest = numpy.clip(empty_nests(new_nest, pa, n, dimension), low_bound, up_bound)


            # Evaluate new solutions and find best
            fnew, best, nest, fitness, evaluations = get_best_nest(nest, new_nest, fitness, n, dimension, objf, evaluations)

            if fnew < fmin:
                fmin = fnew
                bestnest = best
            if len(save_errors_at) and test_error_values and evaluations >= save_errors_at[0]:
                convergence_errors.append(fmin - benchmark.known_optimum_value(func_num, objf))
                save_errors_at.pop(0)
                if test_flags['diversity']:
                    all_diversities[runs][record_result] = testing.diversity_computation(nest, n, dimension)
                    record_result += 1
            if test_flags['movement_in_2d'] and iter < 50:
                testing.plot_contour(filename, nest, low_bound=low_bound, up_bound=up_bound, iteration=iter,
                                     algorithm_name="CS")


        if test_statistics:
            statistics[runs] = numpy.min(fitness)
        if test_error_values:
            all_errors[runs] = numpy.array(convergence_errors)
        print(['CS ' + str(runs) + ': [' + str(fmin) + '] Evaluations: ' + str(
            evaluations) + ' Iterations: ' + str(max_iteration)])

        if best_cuckoo_score > fmin:
            best_cuckoo_score = fmin
            best_cuckoo = bestnest

    if test_statistics:
        statistics = testing.evaluate_all_statistics(statistics)
    if test_error_values:
        filenam = filename + '/cs_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_errors, filenam)
    if test_flags['diversity']:
        filenam = filename + '/CS_diversity_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_diversities, filenam)

    return statistics, best_cuckoo
