import benchmark
import PSO
import CS
import FFA
import BAT
import ctypes
import numpy
import SelfAdaptiveBison
import Bison


problem = {
    'dimension': 10,
    'low_bound': -100,
    'up_bound': 100,
    'function': benchmark.cec2017,
    'func_num': 6, # currently solved problem of the IEEE CEC benchmark testbed
    'population': 50, # NP
    'swarm': 40, # SG swarming group size paraneter
    'elity': 20, # EG elite group size parameter
    'overstep': 3.5,
    'run_support': 0,  # number of iterations for run support strategy
    'filename': 'results/',  # path where to save results
    'boundary_politics': 'hypersphere',
    'self_adaptive': False # possible values: False (normal BIA), True (SA BIA), '7evals' (SA BIA with 7x bigger MaxFES)
}

test_flags = {
    'error_values': True,  # standard IEEE testing
    'statistics': False,
    'movement_in_2d': False,
    'diversity': True,
    'complexity_computation': False,
    'convergence': False,
    'cumulative_movement': False
}

def optimize(func, dim, optimization_algorithm, number_of_runs=51, params_set=1):
    global problem
    problem['func_num'] = func
    problem['dimension'] = dim

    if optimization_algorithm['pso']:
        print("Now dealing with PSO %sD %sF" % (dim, func))
        stats_of_pso, best_pso = PSO.PSO(number_of_runs, problem, test_flags, params_set)
        print("PSO %sD %sF: %s" % (dim, func, best_pso))

    if optimization_algorithm['bison']:
        print("Now dealing with OOP Bison %sD %sF" % (dim, func))
        bison_herd = Bison.BisonAlgorithm(problem, test_flags)
        score, solution = bison_herd.bison_algorithm(number_of_runs)
        print("Bison OOP %sD %sF: %s [%s]" % (dim, func, str(score), str(solution)))

    if optimization_algorithm['sa bison']:
        print("Now dealing with Self Adaptive Bison B %sD %sF" % (dim, func))
        improvements, best_bison = SelfAdaptiveBison.self_adaptive_bison_algorithm(number_of_runs, problem, test_flags)
        print("SA Bison B %sD %sF: %s" % (dim, func, best_bison))

    if optimization_algorithm['cs']:
        print("Now dealing with CS %sD %sF" % (dim, func))
        stats_of_cs, best_cs = CS.CS(number_of_runs, problem, test_flags, params_set)
        print("CS %sD %sF: %s" % (dim, func, best_cs))

    if optimization_algorithm['bat']:
        print("Now dealing with BAT %sD %sF" % (dim, func))
        stats_of_bat, best_bat = BAT.BAT(number_of_runs, problem, test_flags)
        print("BAT %sD %sF: %s" % (dim, func, best_bat))

    if optimization_algorithm['ffa']:
        print("Now dealing with FFA %sD %sF" % (dim, func))
        stats_of_ffa, best_ffa = FFA.FFA(number_of_runs, problem, test_flags, params_set)
        print("FFA %sD %sF: %s" % (dim, func, best_ffa))
    print("Yay!~")


def close_library():
    handle = benchmark.dll_15._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)
    handle = benchmark.dll_17._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)
    handle = benchmark.dll_20._handle  # obtain the DLL handle
    ctypes.windll.kernel32.FreeLibrary(handle)

# Several showcase 2D problems to show the movement in practice
def test_movement(test_scenario=1, saveto='results/movement/'):
    problem['dimension'] = 2
    problem['filename'] = saveto
    problem['overstep'] = 3.5
    problem['population'] = 50
    problem['swarm'] = 40
    problem['elity'] = 20
    problem['run_support'] = 3

    if test_scenario == 1:
        problem['low_bound'] = -1.5
        problem['up_bound'] = 1.5
        problem['function'] = benchmark.rastrigin
    elif test_scenario == 2:
        problem['function'] = benchmark.schwefel
        problem['low_bound'] = -514
        problem['up_bound'] = 514
    elif test_scenario == 3:
        problem['function'] = benchmark.dejong5
        problem['low_bound'] = -65.536
        problem['up_bound'] = 65.536
    elif test_scenario == 4:
        problem['function'] = benchmark.michalewicz
        problem['low_bound'] = 0
        problem['up_bound'] = numpy.pi
    test_flags = {
        'statistics': False,
        'movement_in_2d': True,
        'error_values': False,
        'diversity': False
    }
    bison_herd = Bison.BisonAlgorithm(problem, test_flags)
    score, solution = bison_herd.bison_algorithm(1)
    print("Bison Algorithm movement saved to: %s. Final solution's fitness value: %s. Solution: %s" % (saveto, score, solution))

# Select optimizers to be compared:
optimization_algorithms = {
    'bison': True,
    'sa bison': False,
    'cs': False,
    'pso': False,
    'bat': False,
    'ffa': False
}

# Select a problem to be solved
problem = {
    'dimension': 10,
    'low_bound': -100,
    'up_bound': 100,
    'function': benchmark.cec2017,
    'func_num': 6,      # currently solved problem of the IEEE CEC benchmark testbed
    'population': 50,   # NP
    'swarm': 40,        # SG swarming group size paraneter
    'elity': 20,        # EG elite group size parameter
    'overstep': 3.5,    # overstep parameter
    'run_support': 0,   # number of iterations for run support strategy
    'filename': 'results/', # path where to save results
    'boundary_politics': 'hypersphere',
    'self_adaptive': False # False (to use normal Bison Algorithm), True (to use Self Adaptive Bison Algorithm)
}

optimize(6,10, optimization_algorithms, 5)

# test_movement(2)

close_library()