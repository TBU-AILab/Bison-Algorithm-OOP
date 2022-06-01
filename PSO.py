# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016
within the EvoloPy optimization library
@author: Hossam Faris
-> Modified by Anezka Kazikova to fit the uniform template in 2018
"""

import random
import numpy
import math
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


def PSO(number_of_runs, problem_definition, test_flags, params_set=1):
    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound = problem_definition['up_bound']
    objf = problem_definition['function']
    filename = problem_definition['filename']

    test_error_values = test_flags['error_values']
    test_statistics = test_flags['statistics']
    func_num = problem_definition['func_num']

    statistics = numpy.zeros(number_of_runs)

    # PSO parameters
    if params_set == 2:
        Vmax = 6  #
        PopSize = 30
        wMax = 0.5
        wMin = 0.5
        c1 = 1.9
        c2 = 1.9
    else:
        Vmax = 6  #
        PopSize = 50
        wMax = 0.9
        wMin = 0.2
        c1 = 2
        c2 = 2


    if test_flags['complexity_computation']:
        max_evaluation = 200000
    else:
        max_evaluation = benchmark.get_max_fes(dimension, objf, problem_definition['self_adaptive'])
    max_iteration = round((max_evaluation) / PopSize)
    s = solution()

    all_errors = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))
    all_diversities = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))

    result = numpy.zeros(dimension)
    result_score = float("inf")

    for runs in range(number_of_runs):

        save_errors_at = benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive'])

        ######################## Initializations
        values = numpy.zeros(PopSize)
        evaluations = 0
        vel = numpy.zeros((PopSize, dimension))

        pBestScore = numpy.zeros(PopSize)
        pBestScore.fill(float("inf"))

        pBest = numpy.zeros((PopSize, dimension))
        gBest = numpy.zeros(dimension)

        gBestScore = float("inf")

        pos = numpy.random.uniform(0, 1, (PopSize, dimension)) * (up_bound - low_bound) + low_bound

        convergence_errors = []
        evaluation_curve = numpy.zeros(max_iteration)

        ############################################
        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        record_result = 0

        for l in range(0, max_iteration):

            for i in range(0, PopSize):
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)

                pos[i, :] = numpy.clip(pos[i, :], low_bound, up_bound)
                # zmenit na random a na 40
                # pos[i, :] = numpy.random.uniform(0, 1, (PopSize, dimension)) * (up_bound - low_bound) + low_bound

                # Calculate objective function for each particle
                fitness = objf(pos[i, :], dimension, func_num)
                values[i] = fitness
                evaluations += 1

                if (pBestScore[i] > fitness):
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :]

                if (gBestScore > fitness):
                    gBestScore = fitness
                    gBest = pos[i, :]

            # According to hindawi.com standard sPSO is
            # w = wMin + (max_iteration-l) *  (wMax - wMin) / (max_iteration)
            # Yet EvoloPy uses this formula: w = wMax - l * ((wMax - wMin) / max_iteration);

            # Update the W of PSO
            w = wMax - l * ((wMax - wMin) / max_iteration)  # original EvoloPy

            for i in range(0, PopSize):
                for j in range(0, dimension):
                    r1 = random.random()
                    r2 = random.random()
                    vel[i, j] = w * vel[i, j] + c1 * r1 * (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                    if (vel[i, j] > Vmax):
                        vel[i, j] = Vmax

                    if (vel[i, j] < -Vmax):
                        vel[i, j] = -Vmax

                    pos[i, j] = pos[i, j] + vel[i, j]

            if len(save_errors_at) and test_error_values and evaluations >= save_errors_at[0]:
                all_errors[runs][record_result] = gBestScore - benchmark.known_optimum_value(func_num, objf)
                record_result += 1
                save_errors_at.pop(0)
                if test_flags['diversity']:
                    all_diversities[runs][record_result-1] = testing.diversity_computation(pos, PopSize, dimension)
            if test_flags['movement_in_2d'] and l < 50:
                testing.plot_contour(filename, pos, low_bound=low_bound, up_bound=up_bound, iteration=l,
                                         algorithm_name="PSO")
                # convergence_errors.append(gBestScore - benchmark.known_optimum_value(func_num))

        # if test_error_values:
        # all_errors[runs] = numpy.array(convergence_errors)

        print(['PSO ' + str(runs) + ': [' + str(gBestScore) + '] Evaluations: ' + str(
            evaluations) + ' Iterations: ' + str(max_iteration)])
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart

        # testing.save_progress(s.convergence)
        s.optimizer = "PSO"
        s.objfname = objf.__name__


        if test_statistics:
            statistics[runs] = gBestScore

        if result_score > gBestScore:
            result_score = gBestScore
            result = gBest

    if test_error_values:
        filenam = filename + '/pso_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_errors, filenam)

    # testing.plot_saved_progress(dimension)
    if test_statistics:
        statistics = testing.evaluate_all_statistics(statistics)
    if test_flags['diversity']:
        filenam = filename +  '/PSO_diversity_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_diversities, filenam)
    return statistics, result
