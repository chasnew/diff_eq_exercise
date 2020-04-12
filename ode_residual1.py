# -*- coding: utf-8 -*-

###############################################################################
#                          Libraries to use
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scipy.integrate import odeint
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec


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
    axes.set_title(title)

###############################################################################
#                          ODE fitting functions
###############################################################################

kc=2 #Radioactive decay constant
C0=1 #Initial condition of the model

# General function to solve the ODE model
def GeneralSolver(t, k, C0):
    localK = k
    localC0 = C0

    def ode_model(C, t):
        return -localK * C

    sol = odeint(ode_model, localC0, t)

    return sol[:, 0]


# Solves the ODE model using the initial condition provided above
def solve_ode(t, k):
    return GeneralSolver(t, k, C0)


# Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def make_noisy_data(data, noise):
    return [val + cal for val, cal in zip(data, noise)]

###############################################################################
#                   Data generation for residuals analysis
###############################################################################

# ODE solution
t_data = np.linspace(0, 2)
sol = solve_ode(t_data, kc)

# Generating noise data with mixed signals
white_noise = [np.random.uniform(low=-1, high=1) / 20 for val in sol]
periodic_noise = [np.random.uniform(low=-1, high=1) / 30 + np.sin(val / np.pi) / 30 for val in range(len(t_data))]
linear_noise = [np.random.uniform(low=-1, high=1) / 30 - 0.04 * (val / 30) for val in range(len(t_data))]

###############################################################################
#                            Residuals analysis
###############################################################################

white_signal = make_noisy_data(sol, white_noise)
periodic_signal = make_noisy_data(sol, periodic_noise)
linear_signal = make_noisy_data(sol, linear_noise)

param_white = curve_fit(solve_ode, t_data, white_signal)
param_periodic = curve_fit(solve_ode, t_data, periodic_signal)
param_linear = curve_fit(solve_ode, t_data, linear_signal)

white_solutions = solve_ode(t_data, param_white[0][0])
periodic_solutions = solve_ode(t_data, param_periodic[0][0])
linear_solutions = solve_ode(t_data, param_linear[0][0])

residualsWhite = [val - cal for val, cal in zip(white_signal, white_solutions)]
residualsPeriodic = [val - cal for val, cal in zip(periodic_signal, periodic_solutions)]
residualsLinear = [val - cal for val, cal in zip(linear_signal, linear_solutions)]


###############################################################################
#                    Residual analysis visualization
###############################################################################

def plot_residuals(figure, time, signal, fit_solution, residuals, noise):
    cFig = figure
    gridSp = GridSpec(2, 2)

    ax1 = cFig.add_subplot(gridSp[:, 0])
    ax2 = cFig.add_subplot(gridSp[0, 1])
    ax3 = cFig.add_subplot(gridSp[1, 1])

    ax1.plot(time, signal, 'ro', label='Data')
    ax1.plot(time, fit_solution, label='Regression', path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                                                                  path_effects.Normal()])
    ax1.legend(loc=0)
    PlotStyle(ax1, 'Fitted Model')

    ax2.plot(residuals, path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                                      path_effects.Normal()])
    PlotStyle(ax2, 'Residuals')

    ax3.plot(noise, path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                                  path_effects.Normal()])
    PlotStyle(ax3, 'Noise')

    plt.tight_layout()


fig2 = plt.figure(5, figsize=(13, 5))

plot_residuals(fig2, t_data, white_signal, white_solutions, residualsWhite, white_noise)

fig3 = plt.figure(6, figsize=(13, 5))

plot_residuals(fig3, t_data, periodic_signal, periodic_solutions, residualsPeriodic, periodic_noise)

fig4 = plt.figure(7, figsize=(13, 5))

plot_residuals(fig4, t_data, linear_signal, linear_solutions, residualsLinear, linear_noise)