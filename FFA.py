# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:49:35 2016

@author: hossam
"""

# % ======================================================== %
# % Files of the Matlab programs included in the book:       %
# % Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
# % Second Edition, Luniver Press, (2010).   www.luniver.com %
# % ======================================================== %
#
# % -------------------------------------------------------- %
# % Firefly Algorithm for constrained optimization using     %
# % for the design of a spring (benchmark)                   %
# % by Xin-She Yang (Cambridge University) Copyright @2009   %
# % -------------------------------------------------------- %

import numpy
import math
import time
import testing
import benchmark

func_num = 0


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


def alpha_new(alpha, NGen):
    # % alpha_n=alpha_0(1-delta)^NGen=10^(-4);
    # % alpha_0=0.9
    delta = 1 - (10 ** (-4) / 0.9) ** (1 / NGen);
    alpha = (1 - delta) * alpha
    return alpha


def FFA(number_of_runs, problem_definition, test_flags, params_set=1):
    dimension = problem_definition['dimension']
    low_bound = problem_definition['low_bound']
    up_bound = problem_definition['up_bound']
    objf = problem_definition['function']
    func_num = problem_definition['func_num']
    filename = problem_definition['filename']

    test_statistics = test_flags['statistics']
    test_error_values = test_flags['error_values']
    # General parameters

    n = 50
    if params_set==2:
        n = 20  # number of fireflies

    if test_flags['complexity_computation']:
        max_evaluation = 200000
    else:
        max_evaluation = benchmark.get_max_fes(dimension, objf, problem_definition['self_adaptive'])
    max_iteration = round((max_evaluation) / n)
    all_errors = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))
    evaluations_curve = numpy.zeros(max_iteration)
    statistics = numpy.zeros(number_of_runs)
    all_diversities = numpy.zeros((number_of_runs, len(benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive']))))

    # [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)

    s = solution()

    best_score = float("inf")
    best_pos = numpy.zeros(dimension)

    for runs in range(number_of_runs):

        save_errors_at = benchmark.when_to_record_results(dimension, objf, problem_definition['self_adaptive'])
        record_result = 0
        evaluations = 0

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        # FFA parameters

        if params_set == 2:
            alpha = 0.25
        else:
            alpha = 0.5 # Randomness 0--1 (highly random)
        betamin = 0.20  # minimum value of beta
        gamma = 1.0  # Absorption coefficient

        zn = numpy.ones(n)
        zn.fill(float("inf"))

        # ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
        ns = numpy.random.uniform(0, 1, (n, dimension)) * (up_bound - low_bound) + low_bound
        Lightn = numpy.ones(n)
        Lightn.fill(float("inf"))

        convergence_errors = []

        # Main loop
        for k in range(0, max_iteration):  # start iterations
            # % This line of reducing alpha is optional
            alpha = alpha_new(alpha, max_iteration)

            # % Evaluate new solutions (for all n fireflies)
            for i in range(0, n):
                zn[i] = objf(ns[i, :], dimension, func_num)
                evaluations += 1
                Lightn[i] = zn[i]

            # Ranking fireflies by their light intensity/objectives

            Lightn = numpy.sort(zn)
            Index = numpy.argsort(zn)
            ns = ns[Index, :]

            # Find the current best
            nso = ns
            Lighto = Lightn
            nbest = ns[0, :]
            Lightbest = Lightn[0]

            # % For output only
            fbest = Lightbest

            # % Move all fireflies to the better locations
            #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
            #          Lightbest,alpha,betamin,gamma,Lb,Ub);
            scale = numpy.ones(dimension) * abs(up_bound - low_bound)
            for i in range(0, n):
                # The attractiveness parameter beta=exp(-gamma*r)
                for j in range(0, n):
                    r = numpy.sqrt(numpy.sum((ns[i, :] - ns[j, :]) ** 2));
                    # r=1
                    # Update moves
                    if Lightn[i] > Lighto[j]:  # Brighter and more attractive
                        beta0 = 1
                        beta = (beta0 - betamin) * math.exp(-gamma * r ** 2) + betamin
                        tmpf = alpha * (numpy.random.rand(dimension) - 0.5) * scale
                        ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf

            # ns=numpy.clip(ns, lb, ub)
            IterationNumber = k
            BestQuality = fbest


            if len(save_errors_at) and test_error_values and evaluations >= save_errors_at[0]:
                convergence_errors.append(fbest - benchmark.known_optimum_value(func_num, objf))
                save_errors_at.pop(0)

                if test_flags['diversity']:
                    all_diversities[runs][record_result]=testing.diversity_computation(ns, n, dimension)
                    record_result += 1

            if test_flags['movement_in_2d'] and k < 50:
                testing.plot_contour(filename, ns, low_bound=low_bound, up_bound=up_bound, iteration=k, algorithm_name="FFA")


        if test_statistics:
            statistics[runs] = fbest
        if test_error_values:
            all_errors[runs] = numpy.array(convergence_errors)

        if fbest < best_score:
            best_score = fbest
            best_pos = nbest

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        print(['FFA ' + str(runs) + ': [' + str(fbest) + '] Evaluations: ' + str(
            evaluations) + ' Iterations: ' + str(max_iteration) + ' Time: ' + str(s.executionTime)])

    if test_statistics:
        statistics = testing.evaluate_all_statistics(statistics)
    if test_error_values:
        filenam = filename + '/ffa_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_errors, filenam)
    if test_flags['diversity']:
        filenam = filename + '/FFA_diversity_' + str(func_num) + '_' + str(dimension) + '_param' + str(params_set) + '.csv'
        testing.save_errors_to_file(all_diversities, filenam)
    return best_score, best_pos
