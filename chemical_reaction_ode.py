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
#                              ODE system  solving
###############################################################################

time_steps = np.linspace(0, 20, num=120)

# Chemical reaction model parameters
k1 = 0.3
k2 = 0.25
k3 = 0.1


# Coefficients matrix for model B
# Model B is refered as model02
def make_matrix_model(K1, K2, K3):
    matrix = np.zeros((3, 3))

    matrix[0, 0] = -K1
    matrix[0, 1] = K3

    matrix[1, 0] = K1
    matrix[1, 1] = -(K2 + K3)

    matrix[2, 1] = K2

    return matrix


matrix_02 = make_matrix_model(k1, k2, k3)
initial_conditions = [5, 0, 0]


def kinetic_system(initial_conditions, t):
    return util.make_model(matrix_02, initial_conditions)


system_solution = odeint(kinetic_system, initial_conditions, time_steps)

###############################################################################
#                    Visualization
###############################################################################

plt.figure(1, figsize=(9, 6))

plt.plot(time_steps, system_solution[:, 0], 'b-', label='[A]',
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])
plt.plot(time_steps, system_solution[:, 1], 'g-', label='[B]',
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])
plt.plot(time_steps, system_solution[:, 2], 'm-', label='[C]',
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Concentration', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

ax = plt.gca()
util.plot_style(ax, '')

###############################################################################
#                            Data Generation
###############################################################################

white_noise = [np.random.uniform(low=-1, high=1) / 4 for val in system_solution[:, 2]]
white_signal = util.make_noisy_data(system_solution[:, 2], white_noise)


###############################################################################
#                              ODE fitting
###############################################################################

def solve_model(t, k1, k2, k3, initial_conditions):
    c_k1 = k1
    c_k2 = k2
    c_k3 = k3

    c_init = initial_conditions

    c_matrix = make_matrix_model(c_k1, c_k2, c_k3)

    def LocalModel(cInit, t):
        return util.make_model(c_matrix, cInit)

    solution = odeint(LocalModel, c_init, t)

    return solution[:, 2]


def compute_solution(t, K1, K2, K3):
    return solve_model(t, K1, K2, K3, initial_conditions)


est_parameters = curve_fit(compute_solution, time_steps, white_signal)

f_k1 = est_parameters[0][0]
f_k2 = est_parameters[0][1]
f_k3 = est_parameters[0][2]
print(f'true parameters = [{k1, k2, k3}]')
print(f'estimated parameters = [{f_k1, f_k2, f_k3}]')

fit_solution = compute_solution(time_steps, f_k1, f_k2, f_k3)

###############################################################################
#                        Visualization
###############################################################################

plt.figure(2, figsize=(9, 6))

(markers, stemlines, baseline) = plt.stem(time_steps, white_signal, bottom=0, label='Data', basefmt=" ")
plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5, alpha=0.5)
plt.setp(markers, color="red", alpha=0.75)

solution_label = '[C]'
plt.plot(time_steps, fit_solution, 'm-', label=solution_label,
         path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Concentration', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

plt.ylim(-0.25, 5.2)

ax = plt.gca()
util.plot_style(ax, '')

# Compare estimated solutions with actual solutions

plt.figure(3, figsize=(9, 6))

plt.plot(time_steps, system_solution[:, 0], 'blue', label='[A]')
plt.plot(time_steps, system_solution[:, 1], 'green', label='[B]')
plt.plot(time_steps, system_solution[:, 2], 'magenta', label='[C]')

plt.xlabel('Time', fontsize=16, fontweight='bold')
plt.ylabel('Concentration', fontsize=16, fontweight='bold')
plt.legend(loc=0, fontsize=14)

ax = plt.gca()
util.plot_style(ax, '')

###############################################################################
#                    Residuals Statistical test
###############################################################################

ObRes = [signal - model for signal, model in zip(white_signal, fit_solution)]

KS = stats.ks_2samp(ObRes, white_noise)

print(KS)