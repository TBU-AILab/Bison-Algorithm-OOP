import math
import ctypes
import numpy
import random

name_of_function = " "

dll_17 = ctypes.cdll.LoadLibrary('CEC2017 x64.dll')
# CEC2017 x32.dll for 32bit systems, or CEC2017 x64.dll for 64bit systems
dll_15 = ctypes.cdll.LoadLibrary('CEC2015 x64.dll')
# CEC2015 x32.dll for 32bit systems or CEC2015 x64.dll for 64bit systems
dll_20 = ctypes.cdll.LoadLibrary('CEC2020 x64.dll')
rnd = random.Random()
Z = numpy.linspace(0, 100)


# CEC2020 x32.dll for 32bit systems or CEC2020 x64.dll for 64bit systems

def cec2017(position, dimension, func_num):
    global name_of_function
    name_of_function = 'Cec 2017'
    fun = dll_17.call_function
    fun.restype = ctypes.c_double
    fun.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    result = fun(position, dimension, func_num)
    return result


def cec2015(position, dimension, func_num):
    global name_of_function
    name_of_function = 'Cec 2015'
    fun = dll_15.call_function
    fun.restype = ctypes.c_double
    fun.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    result = fun(position, dimension, func_num)
    return result


def cec2020(position, dimension, func_num):
    global name_of_function
    name_of_function = 'Cec 2020'
    fun = dll_20.call_function
    fun.restype = ctypes.c_double
    fun.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    result = fun(position, dimension, func_num)
    return result


def get_max_fes(dimension, objf=cec2017, self_adaptive=False):
    if self_adaptive == False:
        max_fes = 10000 * dimension
    elif self_adaptive != False:
        max_fes = 10000 * dimension / 7 # because we have 7 population sub groups
    if objf == cec2020:
        if dimension == 5:
            max_fes = 50000
        elif dimension == 10:
            max_fes = 1000000
        elif dimension == 15:
            max_fes = 3000000
        elif dimension == 20:
            max_fes = 10000000
    return max_fes


def when_to_record_results(dimension, objf=cec2017, self_adaptive=False):
    max_fes = get_max_fes(dimension, objf, self_adaptive)
    if objf == cec2020:
        save_result = lambda k: max_fes * dimension ** ((k / 5) - 3)
        save_errors_at = [save_result(0), save_result(1), save_result(2), save_result(3), save_result(4),
                          save_result(5), save_result(6), save_result(7), save_result(8), save_result(9),
                          save_result(10), save_result(11), save_result(12), save_result(13), save_result(14),
                          save_result(15)]
    else:
        save_result = lambda k: max_fes * k
        save_errors_at = [save_result(0.01), save_result(0.02), save_result(0.03), save_result(0.05), save_result(0.1),
                          save_result(0.2), save_result(0.3), save_result(0.4), save_result(0.5), save_result(0.6),
                          save_result(0.7), save_result(0.8), save_result(0.9), save_result(1.0)]
    return save_errors_at


def known_optimum_value(func_num, objf=cec2017):
    if objf == cec2020:
        if func_num == 1:
            return 100
        elif func_num == 2:
            return 1100
        elif func_num == 3:
            return 700
        elif func_num == 4:
            return 1900
        elif func_num == 5:
            return 1700
        elif func_num == 6:
            return 1600
        elif func_num == 7:
            return 2100
        elif func_num == 8:
            return 2200
        elif func_num == 9:
            return 2400
        elif func_num == 10:
            return 2500
    elif objf == cec2015 or objf == cec2017:
        return (func_num * 100)
    else:
        return 0

def rastrigin(position, *args):
    global name_of_function
    name_of_function = 'Rastrigin'
    result = 10 * len(position)
    for x in position:
        # x += 2
        result += (x) ** 2 - 10 * numpy.cos(2 * math.pi * x)
    return result


def dejong1(position, dim=0, func=0):
    global name_of_function
    name_of_function = 'De Jong 1'
    result = 0
    for x in position:
        result += (x) ** 2
    return result


def schwefel(position, *args):
    global name_of_function
    name_of_function = 'Schwefel'
    alpha = 418.982887
    result = 0.0
    for i in range(len(position)):
        result -= position[i] * math.sin(math.sqrt(math.fabs(position[i])))
    return result + alpha * len(position)


def rosenbrock(position, *args):
    global name_of_function
    name_of_function = 'Rosenbrock'
    result = 0.0
    for x in range(1, len(position)):
        result += 100 * (position[x] - position[x - 1] ** 2) ** 2 + (1 - position[x - 1]) ** 2
    return result


def easom(position, *args):
    global name_of_function
    name_of_function = 'Easom'
    result = 0.0
    for x in range(1, len(position)):
        x1 = position[x - 1]
        x2 = position[x]
        exponent = - (x1 - numpy.pi) ** 2 - (x2 - numpy.pi) ** 2
        result += -numpy.cos(x1) * numpy.cos(x2) * numpy.e ** exponent
    return result


# global minimum    for D=2: -1.8013 at [2.20, 1.57]
#                   for D=5: -4.687658
#                   for D=10: -9.66015
# Usually on 0-pi hypercube (visualisation 0-4)
def michalewicz(position, *args):
    global name_of_function
    name_of_function = 'Michalewicz'
    result = 0.0
    for x in range(0, len(position)):
        x1 = position[x]
        m = 10
        result += numpy.sin(x1) * numpy.sin(x*x1*x1/numpy.pi)**(2*m)
    result = result*(-1)
    # 2d simplification:
    x1 = position[0]
    x2 = position[1]
    m = 10
    result = - numpy.sin(x1) * numpy.sin((x1 ** 2) / numpy.pi) ** (2 * m) - (numpy.sin(x2) * numpy.sin((2 * x2 ** 2) / numpy.pi) ** (2 * m))
    return result

# xi ∈ [-65.536, 65.536] (visualisation -50, 50)
def dejong5(position, *args):
    global name_of_function
    name_of_function = 'De Jong 5'
    result = 0.002
    x1=position[0]
    x2=position[1]
    m1 = [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32]
    m2 = [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
    for i in range(0, 25):
        result += 1 / (i + (x1 - m1[i]) ** 6 + (x2 - m2[i]) ** 6)
    result = result ** (-1)

    return result
