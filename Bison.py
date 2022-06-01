# ~~~~~~~~~~~~~~~~~ Bison Algorithm the OOP version ~~~~~~~~~~~~~~~~~
# Swarm optimization algorithm developed by kazikova@utb.cz in 2017


import random
import numpy
import benchmark
import testing


class BisonAlgorithm():
    def __init__(self, problem_definition, test_flag, pop=50, sg=40, eg=20, overstep=3.5, self_adaptive_group=False):
        self.dimension = problem_definition['dimension']
        self.low_bound = problem_definition['low_bound']
        self.up_bound = problem_definition['up_bound']
        self.objf = problem_definition['function']
        self.func_num = problem_definition['func_num']
        self.savefilename = problem_definition['filename']
        self.run_support = problem_definition['run_support'] # parameter defining for many iterations will the swarming group follow a promising solution discovered by the running group
        self.overstep = overstep # parameter defining the maximal possible overstep of the swarmming group
        self.boundary_politics = problem_definition['boundary_politics']
        self.population = pop
        self.swarm_group_size = sg
        self.elite_group_size = eg
        self.max_evaluation = benchmark.get_max_fes(self.dimension, self.objf, problem_definition['self_adaptive'])
        self.max_iteration = round((self.max_evaluation - self.population) / self.population)
        self.bisons = numpy.zeros((self.population, self.dimension), dtype=numpy.double) # population of solutions
        self.bisons_fitness = numpy.zeros(self.population)                               # their fitness values
        self.run_direction = numpy.zeros(self.dimension, dtype=numpy.double)             # direction of the running group
        self.center = numpy.zeros(self.dimension, dtype=numpy.double)                    # center of the elite group
        self.successful_runners = 0 # defines whether right now is the swarming group exploiting discovered running solution or the swarming solution
        self.errors = []            # error values represent how the found solution differs from the ideal solution
        self.test = test_flag

        self.self_adaptive = problem_definition['self_adaptive']
        self.self_adaptive_group = self_adaptive_group
        self.evaluations = 0

        self.initialization()

    def fitness(self, position):
        self.evaluations += 1
        return self.objf(position, self.dimension, self.func_num)

    def check_bounds(self, bison):
        size = self.up_bound - self.low_bound
        if self.boundary_politics == "hypersphere":
            for x in range(self.dimension):
                if bison[x] > self.up_bound:
                    bison[x] = self.low_bound + (abs(bison[x] - self.up_bound) % size)
                elif bison[x] < self.low_bound:
                    bison[x] = self.up_bound - (abs(bison[x] - self.low_bound) % size)
        return bison

    def compute_center(self):
        self.center = numpy.zeros(self.dimension, dtype=numpy.double)
        bison_weight = numpy.ones(self.elite_group_size)

        # There are many ways to compute center. Their impact, however, did not prove to be significant.
        # By default, we use the ranked center computation.
        for x in range(self.elite_group_size):
            bison_weight[x] = (self.elite_group_size - x) * 10
        all_weights = sum(bison_weight)

        for d in range(self.dimension):
            for x in range(self.elite_group_size):
                self.center[d] += (bison_weight[x] * self.bisons[x][d]) / all_weights
        return self.center

    def initialization(self):
        # Initial values generation:~
        neighbourhood = abs(self.up_bound - self.low_bound) / 15
        self.best_solution = numpy.zeros(self.dimension)
        self.best_solution_fitness = numpy.inf

        # position bisons in the swarming group randomly and find the best solution
        for x in range(self.swarm_group_size):
            self.bisons[x] = [random.uniform(self.low_bound, self.up_bound) for i in range(self.dimension)]
            self.bisons_fitness[x] = self.fitness(self.bisons[x])
            if self.best_solution_fitness > self.bisons_fitness[x]:
                self.best_solution_fitness = self.bisons_fitness[x]
                self.best_solution = self.bisons[x]

        # position running bisons around the best solution
        for x in range(self.swarm_group_size, self.population):
            self.bisons[x] = [self.best_solution[i] + random.uniform(-neighbourhood, neighbourhood) for i in
                              range(self.dimension)]
            self.check_bounds(self.bisons[x])
            self.bisons_fitness[x] = self.fitness(self.bisons[x])

        # copy better runners into the swarming group and toss the worse swarming solutions
        sorting_indices = self.bisons_fitness.argsort()
        self.bisons[:self.swarm_group_size] = self.bisons[sorting_indices[:self.swarm_group_size]]
        self.bisons_fitness[:self.swarm_group_size] = self.bisons_fitness[sorting_indices[:self.swarm_group_size]]

        # initiate the run direction vector and results array
        self.run_direction = [random.choice([-1, 1]) * random.uniform(neighbourhood / 3, neighbourhood) for i in
                              range(self.dimension)]

        self.center = self.compute_center()
        return

    def move(self, iteration=0, run=0):
        # subtle alternation of the run direction vector in each iteration
        self.run_direction = [self.run_direction[x] * random.uniform(0.9, 1.1) for x in range(self.dimension)]

        # The Run Support Strategy of the Bison Algorithm works as follows:
        #   If runners find a promising solution, swarmers swarm towards the promising solution
        #   for next few iterations defined by the run support parameter.
        #   Otherwise swarming group swarms towards its center as usual.

        for x, item in enumerate(self.bisons):
            current = numpy.array(self.bisons[x])
            if x < self.swarm_group_size:
                if self.successful_runners > 0:
                    self.swarm(current, 0.95, 1.05)
                else:
                    self.swarm(current, 0, self.overstep)
                current_fitness = self.fitness(current)
                if current_fitness < self.bisons_fitness[x]:
                    self.bisons[x] = current
                    self.bisons_fitness[x] = current_fitness
            if x >= self.swarm_group_size:
                self.run(current)
                self.bisons[x] = current
                self.bisons_fitness[x] = self.fitness(current)

        # Sort the swarming group
        sorting_indices = self.bisons_fitness.argsort()
        # Basic Bison Algorithm:
        self.bisons[:self.swarm_group_size] = self.bisons[sorting_indices[:self.swarm_group_size]]
        self.bisons_fitness[:self.swarm_group_size] = self.bisons_fitness[sorting_indices[:self.swarm_group_size]]

        # Original Bison proposal from Mendel Conference sorted solutions like this:
        # self.bisons = self.bisons[sorting_indices]
        # self.bisons_fitness = self.bisons_fitness[sorting_indices]

        # Check if runners found a promising solution and set appropriate center for next movement
        self.successful_runners -= 1
        runners_found_promising_solution = False
        best_runner_score = self.bisons_fitness[self.swarm_group_size-1]
        best_runner = numpy.copy(self.bisons[self.swarm_group_size-1])
        # Find the best runner & follow it
        for runner in range(self.swarm_group_size, self.population):
            if self.bisons_fitness[runner] < best_runner_score:
                best_runner_score = self.bisons_fitness[runner]
                best_runner = numpy.copy(self.bisons[runner])
                runners_found_promising_solution = True
        if runners_found_promising_solution:
            self.successful_runners = self.run_support
            self.center = numpy.copy(best_runner)
        # Otherwise the center for the swarming movement will be computed from the fittest solutions
        if self.successful_runners <= 0:
            self.center = self.compute_center()


    def swarm(self, bison, from_=0, to_=3.5):
        direction = numpy.zeros(self.dimension, dtype=numpy.double)
        for x in range(self.dimension):
            direction[x] = self.center[x] - bison[x]
            bison[x] += direction[x] * random.uniform(from_, to_)
        self.check_bounds(bison)
        return bison


    def run(self, bison):
        for d in range(self.dimension):
            bison[d] += self.run_direction[d]
        self.check_bounds(bison)
        return bison


    def reset_run(self):
        self.evaluations = 0
        self.max_evaluation = benchmark.get_max_fes(self.dimension, self.objf, self.self_adaptive)
        self.max_iteration = round((self.max_evaluation - self.population) / self.population)
        self.bisons = numpy.zeros((self.population, self.dimension), dtype=numpy.double)
        self.center = numpy.zeros(self.dimension)
        self.errors.clear()


    def bison_algorithm(self, number_of_runs):
        solution_score = 0.0
        solution = numpy.zeros(self.dimension, dtype=numpy.double)
        statistics = numpy.zeros(number_of_runs)
        all_errors = numpy.zeros(
            (number_of_runs,
             len(benchmark.when_to_record_results(self.dimension, self.objf, self.self_adaptive))))
        all_diversities = numpy.zeros(
            (number_of_runs,
             len(benchmark.when_to_record_results(self.dimension, self.objf, self.self_adaptive))))

        for i in range(number_of_runs):

            self.reset_run()
            self.initialization()

            if i == 0:
                solution = numpy.array(self.bisons[0])
                solution_score = self.bisons_fitness[0]
            save_errors_at = benchmark.when_to_record_results(self.dimension, self.objf, self.self_adaptive)
            record_result = 0

            for x in range(self.max_iteration):
                if self.test['movement_in_2d'] and x < 100:
                    testing.plot_contour(self.savefilename, self.bisons, self.center, self.low_bound, self.up_bound, x,
                                         self.elite_group_size, self.swarm_group_size)

                self.move(x, run=i)

                if self.test['error_values'] and len(save_errors_at) > 0 and self.evaluations >= save_errors_at[0]:
                    self.errors.append(
                        self.bisons_fitness[0] - benchmark.known_optimum_value(self.func_num, self.objf))
                    save_errors_at.pop(0)
                    if self.test['diversity']:
                        all_diversities[i][record_result] = testing.diversity_computation(self.bisons, self.population,
                                                                                          self.dimension)
                        record_result += 1


            if self.test['error_values']:
                all_errors[i] = numpy.array(self.errors)
            if self.test['statistics']:
                statistics[i] = self.bisons_fitness[0]
            if solution_score > self.bisons_fitness[0]:
                solution = self.bisons[0]
                solution_score = self.bisons_fitness[0]
            print("Bison Algorithm %s: %s, %s evaluations, %s iterations" %
                  (i, self.bisons_fitness[0], self.evaluations, self.max_iteration))

        # print("Best solution: %s" % solution)
        if self.test['statistics']:
            statistics = testing.evaluate_all_statistics(statistics)
            print("Statistics of bisons: %s" % statistics)
        if self.test['error_values']:
            filename = self.savefilename + '/bison_oop_' + str(self.func_num) + '_' + str(self.dimension) + '.csv'
            testing.save_errors_to_file(all_errors, filename)
        if self.test['diversity']:
            filename = self.savefilename + '/bison_oop_diversity_' + str(self.func_num) + '_' + str(
                self.dimension) + '.csv'
            testing.save_errors_to_file(all_diversities, filename)

        return solution_score, solution
