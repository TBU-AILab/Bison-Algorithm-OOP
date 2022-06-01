import Bison
import benchmark
import numpy
import testing

def self_adaptive_bison_algorithm(number_of_runs, problem, test):

    print("Solving Function %s, Dimension %s with Self Adaptive Bison Algorithm" % (problem["func_num"], problem["dimension"]))

    normal_herd = Bison.BisonAlgorithm(problem, test, pop=50, sg=40, eg=20, overstep=3.5, self_adaptive_group='standard')
    higher_overstep_herd = Bison.BisonAlgorithm(problem, test, 50, 40, 20, 4.5, 'overstep high')
    lower_overstep_herd = Bison.BisonAlgorithm(problem, test, 50, 40, 20, 2.5, 'overstep low')
    higher_sg_herd = Bison.BisonAlgorithm(problem, test, 50, 45, 20, 3.5, 'sg high')
    lower_sg_herd = Bison.BisonAlgorithm(problem, test, 50, 35, 20, 3.5, 'sg low')
    higher_eg_herd = Bison.BisonAlgorithm(problem, test, 50, 40, 25, 3.5, 'eg high')
    lower_eg_herd = Bison.BisonAlgorithm(problem, test, 50, 40, 15, 3.5, 'eg low')

    parameters = numpy.zeros((3, number_of_runs, normal_herd.max_iteration))

    normal_herd.reset_run()

    solution_score = 0.0
    solution = numpy.zeros(normal_herd.dimension, dtype=numpy.double)
    statistics = numpy.zeros(number_of_runs)
    all_best_pop_counters = numpy.zeros((number_of_runs,7))
    all_errors = numpy.zeros(
        (number_of_runs,
         len(benchmark.when_to_record_results(normal_herd.dimension, normal_herd.objf, self_adaptive=normal_herd.self_adaptive))))
    all_diversities = numpy.zeros(
        (number_of_runs,
         len(benchmark.when_to_record_results(normal_herd.dimension, normal_herd.objf, self_adaptive=normal_herd.self_adaptive))))

    all_herds = [normal_herd, lower_overstep_herd, higher_overstep_herd, lower_sg_herd, higher_sg_herd, lower_eg_herd, higher_eg_herd]

    for i in range(number_of_runs):

        counter_of_best_population = {
            "core": 0,
            "sg high": 0,
            "sg low": 0,
            "eg high": 0,
            "eg low": 0,
            "overstep high": 0,
            "overstep low": 0
        }

        for herd in all_herds:
            herd.reset_run()
            herd.initialization()

        if i == 0:
            solution = numpy.array(normal_herd.bisons[0])
            solution_score = normal_herd.bisons_fitness[0]
        save_errors_at = benchmark.when_to_record_results(normal_herd.dimension, normal_herd.objf, self_adaptive=normal_herd.self_adaptive)
        record_result = 0
        self_adaptation_count = 0
        x=0

        for x in range(normal_herd.max_iteration):

            parameters[0][i][x] = normal_herd.overstep
            parameters[1][i][x] = normal_herd.swarm_group_size
            parameters[2][i][x] = normal_herd.elite_group_size

            if test['movement_in_2d'] and x < 50:
                testing.plot_contour(normal_herd.savefilename, normal_herd.bisons, normal_herd.center, normal_herd.low_bound, normal_herd.up_bound, x,
                                     normal_herd.elite_group_size, normal_herd.swarm_group_size)

            for herd in all_herds:
                if x < herd.max_iteration:
                    herd.move(x)

            # self adaptation of parameters:

            best_herd = min(all_herds, key=lambda x: x.bisons_fitness[0])
            if best_herd.bisons_fitness[0] < normal_herd.bisons_fitness[0]:
                normal_herd.elite_group_size = best_herd.elite_group_size
                normal_herd.overstep = best_herd.overstep
                normal_herd.swarm_group_size = best_herd.swarm_group_size
                normal_herd.bisons[normal_herd.swarm_group_size-1] = best_herd.bisons[0]
                normal_herd.bisons_fitness[normal_herd.swarm_group_size-1] = best_herd.bisons_fitness[0]
                self_adaptation_count += 1

                if best_herd.self_adaptive_group == 'overstep high':
                    best_herd.overstep += 0.01
                elif best_herd.self_adaptive_group == 'overstep low':
                    best_herd.overstep -= 0.01
                if best_herd.self_adaptive_group == 'sg high':
                    best_herd.swarm_group_size += 1
                elif best_herd.self_adaptive_group == 'sg low':
                    best_herd.swarm_group_size -= 1
                if best_herd.self_adaptive_group == 'eg high':
                    best_herd.elite_group_size += 1
                elif best_herd.self_adaptive_group == 'eg low':
                    best_herd.elite_group_size -= 1

                best_herd.overstep = numpy.clip(best_herd.overstep, 1, 14)
                best_herd.elite_group_size = numpy.clip(best_herd.elite_group_size, 1, best_herd.swarm_group_size)
                best_herd.swarm_group_size = numpy.clip(best_herd.swarm_group_size, best_herd.elite_group_size,
                                                            best_herd.population-1)
                counter_of_best_population[best_herd.self_adaptive_group] += 1

            elif best_herd == normal_herd and self_adaptation_count > 0:
                counter_of_best_population["core"] += 1
                for herd in all_herds:
                    herd.overstep = numpy.clip(herd.overstep, 1, 14)
                    herd.swarm_group_size = numpy.clip(herd.swarm_group_size, herd.elite_group_size,
                                                herd.population)
                    herd.elite_group_size = numpy.clip(herd.elite_group_size, 1, herd.swarm_group_size)

            if test['error_values'] and ((len(save_errors_at) > 0 and normal_herd.evaluations >= save_errors_at[0]) or x==normal_herd.max_iteration-1):
                normal_herd.errors.append(
                    normal_herd.bisons_fitness[0] - benchmark.known_optimum_value(normal_herd.func_num, normal_herd.objf))
                save_errors_at.pop(0)
                if test['diversity']:
                    all_diversities[i][record_result] = testing.diversity_computation(normal_herd.bisons, normal_herd.population,
                                                                                      normal_herd.dimension)
                    record_result += 1

        if test['error_values']:
            all_errors[i] = numpy.array(normal_herd.errors)

            all_best_pop_counters[i][0] = numpy.array(counter_of_best_population["core"])
            all_best_pop_counters[i][1] = numpy.array(counter_of_best_population["sg high"])
            all_best_pop_counters[i][2] = numpy.array(counter_of_best_population["sg low"])
            all_best_pop_counters[i][3] = numpy.array(counter_of_best_population["eg high"])
            all_best_pop_counters[i][4] = numpy.array(counter_of_best_population["eg low"])
            all_best_pop_counters[i][5] = numpy.array(counter_of_best_population["overstep high"])
            all_best_pop_counters[i][6] = numpy.array(counter_of_best_population["overstep low"])

        if solution_score > normal_herd.bisons_fitness[0]:
            solution = normal_herd.bisons[0]
            solution_score = normal_herd.bisons_fitness[0]
        print("Bison Algorithm %s: %s, %s evaluations/%s max evaluations, %s iterations, overstep %s, SG %s, EG %s, self adaptation %s" %
              (i, normal_herd.bisons_fitness[0], normal_herd.evaluations, normal_herd.max_evaluation, normal_herd.max_iteration, normal_herd.overstep, normal_herd.swarm_group_size, normal_herd.elite_group_size, self_adaptation_count))
        print("Best population statistics: C:%s SG h:%s SG l:%s EG h:%s EG l:%s OV h:%s OV l:%s" % (counter_of_best_population["core"],
                                                                                                    counter_of_best_population["sg high"],
                                                                                                    counter_of_best_population["sg low"],
                                                                                                    counter_of_best_population["eg high"],
                                                                                                    counter_of_best_population["eg low"],
                                                                                                    counter_of_best_population["overstep high"],
                                                                                                    counter_of_best_population["overstep low"]))

    if test['statistics']:
        statistics = testing.evaluate_all_statistics(statistics)
        print("Statistics of bisons: %s" % statistics)
    if test['error_values']:
        filename = normal_herd.savefilename + '/sa_bison_oop_' + str(normal_herd.func_num) + '_' + str(normal_herd.dimension) + '.csv'
        testing.save_errors_to_file(all_errors, filename)
        filename = normal_herd.savefilename + '/sa_oversteps_' + str(normal_herd.func_num) + 'F_' + str(normal_herd.dimension) + 'D.csv'
        testing.save_errors_to_file(parameters[0], filename)
        filename = normal_herd.savefilename + '/sa_sg_' + str(normal_herd.func_num) + 'F_' + str(normal_herd.dimension) + 'D.csv'
        testing.save_errors_to_file(parameters[1], filename)
        filename = normal_herd.savefilename + '/sa_eg_' + str(normal_herd.func_num) + 'F_' + str(
            normal_herd.dimension) + 'D.csv'
        testing.save_errors_to_file(parameters[2], filename)
        filename = normal_herd.savefilename + '/best_pop_counter_' + str(normal_herd.func_num) + 'F_' + str(
            normal_herd.dimension) + 'D.csv'
        testing.save_errors_to_file(all_best_pop_counters, filename)
    if test['diversity']:
        filename = normal_herd.savefilename + '/sa_bison_oop_diversity_' + str(normal_herd.func_num) + '_' + str(normal_herd.dimension) + '.csv'
        testing.save_errors_to_file(all_diversities, filename)

    return solution_score, solution