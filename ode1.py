import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scipy.integrate import odeint
from scipy.optimize import curve_fit

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
#                                 ODE solver
###############################################################################

kc=2 #Radioactive decay constant
C0=1 #Initial condition of the model

# Numpy array that contains the integration times
solver_time = np.linspace(0, 2)

def ODE(C, t):
    return -kc * C

# Solution of the model given the initial condition using a method similar to Euler's method
model_solution = odeint(ODE, C0, solver_time)

###############################################################################
#                          ODE fitting functions
###############################################################################

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
#                         ODE fitting visualization
###############################################################################

# Solving the ODE model
t_vals = np.linspace(0, 2, num=1000)
solution = solve_ode(t_vals, kc)

# Making some simulated data to perform regression
white_noise = [np.random.uniform(low=-1, high=1) / 20 for val in solution]
white_signal = make_noisy_data(solution, white_noise)
Kp = curve_fit(solve_ode, t_vals, white_signal)[0][0]

# Parameter estimation
fit_solution = solve_ode(t_vals, Kp)

plt.figure(2)

plt.plot(t_vals, white_signal, 'ro', label='Data')
plt.plot(t_vals, fit_solution, label='Regression', path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                                                                 path_effects.Normal()])
ax = plt.gca()
ax.legend(loc=0)
PlotStyle(ax, '')