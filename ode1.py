import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scipy.integrate import odeint
from scipy.optimize import curve_fit

import util

plt.style.use('seaborn')

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
def solve_ode(t, k, C0):
    localK = k
    localC0 = C0

    def ode_model(C, t):
        return -localK * C

    sol = odeint(ode_model, localC0, t)

    return sol[:, 0]


# Solves the ODE model using the initial condition provided above
def compute_solution(t, k):
    return solve_ode(t, k, C0)


###############################################################################
#                         ODE fitting visualization
###############################################################################

# Solving the ODE model
t_vals = np.linspace(0, 2, num=1000)
solution = compute_solution(t_vals, kc)

# Making some simulated data to perform regression
white_noise = [np.random.uniform(low=-1, high=1) / 20 for val in solution]
white_signal = util.make_noisy_data(solution, white_noise)
Kp = curve_fit(compute_solution, t_vals, white_signal)[0][0]

# Parameter estimation
fit_solution = compute_solution(t_vals, Kp)

plt.figure(2)

plt.plot(t_vals, white_signal, 'ro', label='Data')
plt.plot(t_vals, fit_solution, label='Regression', path_effects=[path_effects.SimpleLineShadow(alpha=0.2, rho=0.2),
                                                                 path_effects.Normal()])
ax = plt.gca()
ax.legend(loc=0)
util.plot_style(ax, '')