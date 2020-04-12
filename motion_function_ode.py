# -*- coding: utf-8 -*-

###############################################################################
#                          Libraries to use
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scipy.integrate import odeint
from scipy.optimize import curve_fit

import scipy.stats as stats


###############################################################################
#                    General plot functions
###############################################################################

plt.style.use('seaborn')

# Elimates the left and top lines and ticks in a matplotlib plot
def PlotStyle(axes, title):
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    axes.xaxis.set_tick_params(labelsize=14)
    axes.yaxis.set_tick_params(labelsize=14)
    axes.set_title(title)


###############################################################################
#                    General Model Construction
###############################################################################

# Performs the dot product to make the model/construct system of differential equations
def make_model(coef_matrix, init_conditions):
    return np.dot(coef_matrix, init_conditions)


###############################################################################
#                              ODE system solving
###############################################################################

time_steps = np.linspace(0, 20, num=150)

# Parameters for Model A
alpha = 0.4
beta = 1


# Matrix of coeficients for model A
# Model A is refered in this script as model 01
def make_matrix_model(alpha, beta):
    matrix = np.zeros((2, 2))

    matrix[0, 0] = alpha
    matrix[0, 1] = -beta
    matrix[1, 0] = 1

    return matrix


# Integrating Model A
matrix01 = make_matrix_model(alpha, beta)
init_conds = np.array([1, 1])


def SODE(initial_conditions, t):
    return make_model(matrix01, initial_conditions)


solution = odeint(SODE, init_conds, time_steps)

###############################################################################
#                    Visualisation
###############################################################################

derivative_label = r'$\dfrac{d}{dt} f(t) $'
solution_label = r'$f(t)$'

plt.figure(1, figsize=(9, 6))

plt.plot(time_steps, solution[:, 1], 'b-', label=solution_label,
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])
plt.plot(time_steps, solution[:, 0], 'g-', label=derivative_label,
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Displacement', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

ax = plt.gca()
PlotStyle(ax, '')


###############################################################################
#                        Data Generation
###############################################################################

# Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def MakeNoisyData(Data, Noise):
    return [val + cal for val, cal in zip(Data, Noise)]


WhiteNoise = [np.random.uniform(low=-1, high=1) * 3 for val in Solution[:, 1]]
WhiteSignal = MakeNoisyData(Solution[:, 1], WhiteNoise)


###############################################################################
#                              ODE fitting
###############################################################################

# Function for parameter estimation
def solve_model(t, Alpha, Beta, InitialConditions):
    cAlpha = Alpha
    cBeta = Beta
    cInit = InitialConditions

    cMatrix = MakeModelMatrix01(cAlpha, cBeta)

    def LocalModel(cInit, t):
        return MakeModel(cMatrix, cInit)

    Solution = odeint(LocalModel, cInit, t)

    return Solution[:, 1]


def ModelSolution01(t, Alpha, Beta):
    return ModelSolver01(t, Alpha, Beta, Int)


Model01Params = curve_fit(solve_model, SolverTime, WhiteSignal)

###############################################################################
#                    Fit solution
###############################################################################

fAlpha = Model01Params[0][0]
fBeta = Model01Params[0][1]

FitSolutionA = ModelSolution01(SolverTime, fAlpha, fBeta)

###############################################################################
#                    Visualization
###############################################################################

plt.figure(2, figsize=(9, 6))

(markers, stemlines, baseline) = plt.stem(SolverTime, WhiteSignal, bottom=-42, label='Data', basefmt=" ")
plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5, alpha=0.5)
plt.setp(markers, color="red", alpha=0.75)

plt.plot(SolverTime, FitSolutionA, 'b-', label=SolutionLabel,
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Displacement', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

plt.ylim(-42, 75)

ax = plt.gca()
PlotStyle(ax, '')

###############################################################################
#                    Residuals Statistical test
###############################################################################

ObRes = [signal - model for signal, model in zip(WhiteSignal, FitSolutionA)]

KS = stats.ks_2samp(ObRes, WhiteNoise)

print(KS)