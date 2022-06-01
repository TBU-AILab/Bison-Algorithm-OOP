# Bison-Algorithm-OOP

the object-oriented version with the Run Support Strategy & the self-adaptive variant

The Bison Algorithm is an bio-inspired metaheuristic optimization algorithm that you are welcome to use!

created by Anezka Kazikova in 2018-2022
feel free to contact me at kazikova@utb.cz
Tomas Bata University in Zlin, Czech Republic


Citation: Kazikova A., Pluhacek M., Kadavy T., Senkerik R. (2020) Introducing the Run Support Strategy for the Bison Algorithm. In: Zelinka I., Brandstetter P., Trong Dao T., Hoang Duy V., Kim S. (eds) AETA 2018 - Recent Advances in Electrical Engineering and Related Sciences: Theory and Application. AETA 2018. Lecture Notes in Electrical Engineering, vol 554. Springer, Cham

There is also a non-object oriented version on https://github.com/TBU-AILab/Bison-Algorithm
Both versions are equally functional, the first one is simpler. This version, howerver, has more options and possibilities. And the Self-Adaptive modification as a bonus.


FILES

- Bison.py > source code of the Bison Algorithm class with the Run Support Strategy. 
 	- The main function is the bison_algorithm(number_of_runs). Use as follows:
        - bison_herd = Bison.BisonAlgorithm(problem, test_flags)
        - score, solution = bison_herd.bison_algorithm(number_of_runs)
        - Example of usage is in compare.py
- benchmark.py > here you can write your own objective functions to optimize. It includes implementation of the IEEE CEC 2015, 2017 and 2020 benchmark libraries, and some other functions like Easom, Schwefel, etc.
- compare.py > main executable code. Can compare more algorithms, or run only one. Define the problem in dictionary 'problem' and compared optimization algorithms in dictionary 'optimization_algorithm'.
- testing.py > support functions for movement visualization and saving files.
- PSO.py > Particle Swarm Optimization algorithm for comparison. Based on the EvoloPy library, modified for the use of this code.
- CS.py > Cuckoo Search algorithm for comparison. Based on the EvoloPy library, modified for the use of this code.
- BAT.py > Bat Algorithm algorithm for comparison. Based on the EvoloPy library, modified for the use of this code.
- FFA.py > Firefly Algorithm algorithm for comparison. Based on the EvoloPy library, modified for the use of this code. - Be aware, that this algorithm is extremely slow.
- SelfAdaptiveBison.py > bonus code of the Self Adaptive modification of the Bison Algorithm. The main function is: self_adaptive_bison_algorithm(number_of_runs, problem, test_flag).
        This modification work on the sub population principles - exploits various parameter configurations, and copies the ones with best results to one core population.
        Provides interesting information about the inner dynamics of the Bison Algorithm.
        However, for normal optimization purposes, I would still suggest the basic Bison Algorithm in Bison.py

If PSO, BAT, FFA or CS used, please, cite: Faris, Hossam & Aljarah, Ibrahim & Mirjalili, Seyedali & Castillo, Pedro & Merelo Guervós, Juan. (2016). EvoloPy: An Open-Source Nature-Inspired Optimization Framework in Python. 10.5220/0006048201710177.

HOW TO USE THE ALGORITHM FOR OPTIMIZATION?

In compare.py:
- define optimized problem 
- pick optimization algorithm(s)
- choose, what you want to test in test_flags
- for optimization us function optimize(x, dimension, optimization_algorithm)
	where x = number of tested function in CEC benchmark
	dimension = dimensionality (CEC has only 10D, 30D, 50D, 100D)
	optimization_algorithm copy as is, defines which algorithm to use
- for pictures of 2D movement, use function test_movement()


VERSION OF PYTHON AND LIBRARIES
- Developed for Python 3.6.0
- With libraries: 
-	NumPy 1.12.0, 
-	MatPlotLib 2.0.0


With the wish of many great optimization successes,

Yours sincerely,

Anezka Kazikova
