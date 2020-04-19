import numpy as np

# Elimates the left and top lines and ticks in a matplotlib plot
def plot_style(axes, title):
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
#                          Data generation
###############################################################################

# Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def make_noisy_data(data, noise):
    return [val + cal for val, cal in zip(data, noise)]