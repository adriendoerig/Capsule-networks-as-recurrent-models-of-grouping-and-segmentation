"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets

Create the final performance plots
@author: Lynn Schmittwilken
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from parameters import parameters

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')


def plot_uncrowding_results(results, error, parameters, save=None):
    '''
    Visualize the final results of the network's vernier discrimation performance
    
    Parameters
    ----------
    results: list of floats
             Network's vernier accuracy outputs
    error: list of floats
           Network's standard error for vernier accuracy
    parameters: flags
                Contains all parameters defined in parameters.py
    save: str
          if None do not save the figure, else provide a data path
    '''
    N = len(results)
    ind = np.arange(N)

    fig, ax = plt.subplots(figsize=(20,10))
    n_categories = len(parameters.test_data_paths)
    ax.bar(ind[0:n_categories], results[0:n_categories], yerr=error[0:n_categories],
           align='center', alpha=0.5, ecolor='black', capsize=4)
    ax.bar(ind[n_categories:], results[n_categories:], yerr=error[n_categories:],
           align='center', alpha=0.5, ecolor='black', capsize=4)

    # Add some text for labels, title and axes ticks, and save figure
    ax.set_ylabel('Performance gain in %correct')
    ax.set_title('Vernier decoder performance')
    ax.set_xticks([])
    ax.set_xticklabels([])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        plt.close()


# How many networks have been trained?
n_iterations = parameters.n_iterations

# The total number of training steps is equal to n_steps*n_rounds
# (e.g. if n_steps=1000 and n_rounds=2, the network performance will be tested
#  using the test set after 1000 and 2000 steps)
n_rounds = parameters.n_rounds

# Number of test conditions used
n_idx = parameters.n_idx

# Total number of different shape types used
n_categories = len(parameters.test_crowding_data_paths)

# Initialize results
results = np.zeros(shape=(n_categories, n_idx, n_iterations))

# Get final results from .txt files
for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'

    for n_category in range(n_categories):
        txt_file_name = log_dir + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
        '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

        with open(txt_file_name, 'r') as f:
            lines = f.read()
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
            numbers = np.float32(numbers)
            # Since the categories 4stars and 6stars involve numbers, we get
            # rid of these numbers
            numbers = numbers[numbers!=4]
            numbers = numbers[numbers!=6]
            results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


# We want to plot the final results as difference between the uncrowding and
# no-uncrowding condition
diff_results = np.squeeze(results[:, 2, :] - results[:, 1, :])
mean_diff_results = np.squeeze(np.mean(diff_results, 1)) * 100
error_diff_results = np.squeeze(np.std(diff_results, 1)) * 100 / np.sqrt(n_iterations)


# Create and save the final performance figure:
performance_png_file = parameters.logdir + '/performance_plots.png'
plot_uncrowding_results(mean_diff_results, error_diff_results, performance_png_file)

print('... Finished creation of /performance_plots.png')
print('-------------------------------------------------------')
