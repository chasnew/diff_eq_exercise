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

import util

plt.style.use('seaborn')

###############################################################################
#                              ODE system solving
###############################################################################

time_steps = np.linspace(0, 20, num=150)

# Parameters for Model A
alpha = 0.4
beta = 1


# Matrix of coefficients for model A
# Model of damped motion of an object (./img/damped_motion_equation_system.png)
def make_matrix_model(alpha, beta):
    matrix = np.zeros((2, 2))

    matrix[0, 0] = -alpha
    matrix[0, 1] = -beta
    matrix[1, 0] = 1

    return matrix


# Integrating Model A
matrix01 = make_matrix_model(alpha, beta)
init_conds = np.array([1, 1])


def SODE(initial_conditions, t):
    return util.make_model(matrix01, initial_conditions)

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
util.plot_style(ax, '')


###############################################################################
#                        Data Generation
###############################################################################

white_noise = [np.random.uniform(low=-0.1, high=0.1) * 3 for val in solution[:, 1]]
white_signal = util.make_noisy_data(solution[:, 1], white_noise)


###############################################################################
#                              ODE fitting
###############################################################################

# Function for parameter estimation
def solve_model01(t, alpha, beta, init_conditions):
    c_alpha = alpha
    c_beta = beta
    c_init = init_conditions

    c_matrix = make_matrix_model(c_alpha, c_beta)

    def local_ode_model(c_init, t):
        return util.make_model(c_matrix, c_init)

    solution = odeint(local_ode_model, c_init, t)

    return solution[:, 1] # return only one dimension of the ode solution


def compute_solution(t, alpha, beta):
    return solve_model01(t, alpha, beta, init_conds) # call a function to solve ODE


est_params = curve_fit(compute_solution, time_steps, white_signal)

###############################################################################
#                    Fit solution
###############################################################################

f_alpha = est_params[0][0]
f_beta = est_params[0][1]
print(f'true alpha = {alpha}, true beta = {beta}')
print(f'estimated alpha = {f_alpha}, estimated beta = {f_beta}')

fit_solution = compute_solution(time_steps, f_alpha, f_beta)

###############################################################################
#                    Visualization
###############################################################################

plt.figure(2, figsize=(9, 6))

(markers, stemlines, baseline) = plt.stem(time_steps, white_signal, bottom=-1.5, label='Data', basefmt=" ")
plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5, alpha=0.5)
plt.setp(markers, color="red", alpha=0.75)

plt.plot(time_steps, fit_solution, 'b-', label=solution_label,
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Displacement', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

plt.ylim(-1.5, 1.5)

ax = plt.gca()
util.plot_style(ax, '')

###############################################################################
#                    Residuals Statistical test
###############################################################################

ObRes = [signal - model for signal, model in zip(white_signal, fit_solution)]

KS = stats.ks_2samp(ObRes, white_noise)

print(KS)