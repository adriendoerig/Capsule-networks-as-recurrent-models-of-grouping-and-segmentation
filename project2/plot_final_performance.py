"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 2: The role of recurrent processing

First, clean the results by discarding all networks that have floored or ceiled
performances.
Then, statistically analyze whether the increase in performance (more precisely:
the performance slopes) with increasing time/routing iterations are significantly
different between the line category and the cuboid category or each condition
vs. zero.
Finally, create the final performance plots for the model and the human experiment.

@author: Lynn Schmittwilken
"""

import re
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from parameters import parameters


print('-------------------------------------------------------')
print('Start the cleaning and evaluation of the results ...')
print('-------------------------------------------------------')

###########################
#      Preparations:      #
###########################
# Choose whether to plot the errors or the improvements in performance
plot_type = 'improvement'  # threshold-like, improvement
save_plots = True

# How many networks have been trained?
n_iterations = parameters.n_iterations

# Number of test conditions used
n_idx = parameters.n_idx

# The total number of training steps is equal to n_steps*n_rounds
# (e.g. if n_steps=1000 and n_rounds=2, the network performance will be tested
#  using the test set after 1000 and 2000 steps)
n_rounds = parameters.n_rounds

# Total number of different shape types used
n_categories = len(parameters.test_crowding_data_paths)

# Minimum and maximum number of routing iterations
routing_min = parameters.routing_min
routing_max = parameters.routing_max

# The performance for this routing iteration will be used to evaluate whether
# the performance is floored or ceiled
chosen_routing_iter = parameters.train_iter_routing

# Initialize results
results = np.zeros(shape=(n_iterations, n_categories, n_idx, routing_max-routing_min+1))

# Get final results from .txt files
for idx_routing in range(routing_min, routing_max+1):
    idx_routing_idx = idx_routing-routing_min

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                results[idx_execution, :, :, idx_routing_idx] = np.reshape(numbers, [-1, n_idx])


###########################
#      Data cleaning:     #
###########################
# We have to clean the results from ceiled or floored performances (especially
# from performances below 50%):
# 1. if the max performance for the flankers+vernier condition is smaller than
#    0.55 (= floored)
# 2. if the min performance for the vernier-alone or flankers+vernier condition
#    is smaller than 0.5 (= floored and misleading)
# 3. if the mean performance for the vernier-alone and flankers+vernier condition
#    is larger than 0.95 (= ceiled)
crit_value1 = np.max(np.squeeze(results[:, :, 1, chosen_routing_iter-routing_min]), 1)
crit_idx1 = np.where(crit_value1 < 0.55)

crit_value2 = np.min(results[:, 0:2, 1, chosen_routing_iter-routing_min], 1)
crit_idx2 = np.where(crit_value2 < 0.45)

crit_value3 = np.mean(results[:, 0:2, 1, chosen_routing_iter-routing_min], 1)
crit_idx3 = np.where(crit_value3 > 0.95)

crit_idx_all = np.unique(np.concatenate((crit_idx1, crit_idx2, crit_idx3), 1))
good_idx_all = np.delete(np.arange(0, n_iterations), crit_idx_all)

cleaned_results = results[good_idx_all, :, :, :]


###########################
#         Saving:         #
###########################
for idx_routing in range(routing_min, routing_max+1):
    idx_routing_idx = idx_routing-routing_min
    
    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_mean_cleaned_iterations_' + str(len(good_idx_all)) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results_mean = np.mean(cleaned_results[:, :, :, idx_routing_idx], 0)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')

    # Saving final stds:
    final_result_file = parameters.logdir + '/final_results_std_cleaned_iterations_' + str(len(good_idx_all)) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results_std = np.std(cleaned_results[:, :, :, idx_routing_idx], 0)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')


###########################
#      Data analysis:     #
###########################
# Number of networks left after cleaning
n_networks = len(good_idx_all)

# Create variable for cleaned model results
model_results = cleaned_results

# Create variable for human results
human_results = np.array([[[1595.7, 1088.4, 1549.4, 1143.1, 1221.6, 1590.7, 1139.7], [1556.9, 974.1, 1567.3, 654.8, 729.0, 593.6, 456.2]],  # n_subj, [lines, cuboids], n_datapoints
                          [[1399.4, 700.2, 941.5, 732.0, 1184.1, 368.3, 717.5],      [1423.9, 837.7, 717.0, 314.7, 248.5, 73.9, 154.2]],
                          [[2000.0, 1848.6, 2000.0, 2000.0, 1676.1, 1762.6, 934.7],  [2000.0, 2000.0, 2000.0, 2000.0, 922.7, 758.2, 384.8]],
                          [[1579.1, 1415.0, 1699.6, 1432.0, 1983.0, 1206.8, 1588.2], [2000.0, 981.3, 1650.2, 288.4, 525.5, 283.6, 398.7]],
                          [[834.4, 749.9, 935.6, 830.8, 849.7, 495.1, 526.5],        [1024.2, 1104.7, 927.6, 491.8, 271.4, 108.8, 168.8]]])


###########################
#     Fit lin functs:     #
###########################
# Fit a line to match the performance values
def flin(var, a, b):
    return a * var + b

model_coeffs = np.zeros((n_networks, n_categories, 2))
human_coeffs = np.zeros((5, 2, 2))  # n_subj, [lines, cuboids], [a,b]

chosen_idx = 1
# Model
for this_network in range(n_networks):
    for this_category in range(n_categories):
        these_results = model_results[this_network, this_category, chosen_idx, :]
        model_coeffs[this_network, this_category, :], _ = curve_fit(flin, np.array(range(routing_min, routing_max+1)), these_results, method='dogbox', p0=(0, np.min(these_results)))
        check = 0
        if check:
            x = np.linspace(0, routing_max+2, 100)
            plt.figure()
            plt.plot(x, flin(x, *model_coeffs[this_network, this_category, :]), 'b')
            plt.plot(range(routing_min, routing_max+1), these_results, 'ro')
            plt.title('Net, category: ' + str((this_network, this_category)) + ' a = ' + str(model_coeffs[this_network, this_category, 0]) + ', b = ' + str(model_coeffs[this_network, this_category, 1]))
            plt.show()

# Humans
for this_human in range(human_results.shape[0]):
    for this_category in range(human_results.shape[1]):
        these_results = human_results[this_human, this_category, :]
        human_coeffs[this_human, this_category, :], _ = curve_fit(flin, np.array(range(1, 8)), these_results, method='dogbox', p0=(0, np.min(these_results)))
        check = 0
        if check:
            x = np.linspace(0, routing_max + 2, 100)
            plt.figure()
            plt.plot(x, flin(x, *human_coeffs[this_human, this_category, :]), 'b')
            plt.plot(range(1, 8), these_results, 'ro')
            plt.title('Subj, category: ' + str((this_human, this_category)) + ' a = ' + str(human_coeffs[this_human, this_category, 0]) + ', b = ' + str(human_coeffs[this_human, this_category, 1]))
            plt.show()


#############################
#   Statistical analysis:   #
############################
# Compare the performance slopes with t-tests
model_slopes = model_coeffs[:,:,0]
human_slopes = human_coeffs[:,:,0]

# Humans:
# Lines vs. cuboids
human_lines_vs_cuboids_t_stat, human_lines_vs_cuboids_p_value = stats.ttest_rel(human_slopes[:,0], human_slopes[:,1])
# Lines vs. zero
human_lines_vs_zero_t_stat, human_lines_vs_zero_p_value = stats.ttest_1samp(human_slopes[:,0], 0)
# Cuboids vs. zero
human_cuboids_vs_zero_t_stat, human_cuboids_vs_zero_p_value = stats.ttest_1samp(human_slopes[:,1], 0)

# Model:
# Lines vs. cuboids
model_lines_vs_cuboids_t_stat, model_lines_vs_cuboids_p_value = stats.ttest_rel(model_slopes[:,0], model_slopes[:,1])
# Lines vs. zero
model_lines_vs_zero_t_stat, model_lines_vs_zero_p_value = stats.ttest_1samp(model_slopes[:,0], 0)
# Cuboids vs. zero
model_cuboids_vs_zero_t_stat, model_cuboids_vs_zero_p_value = stats.ttest_1samp(model_slopes[:,1], 0)


# Write results of statistical analysis to .txt file
txt_file = parameters.logdir + '/final_performance_statistics.txt'
with open(txt_file, 'w') as f_txt:
    text = ['-------- HUMANS --------',
            '\nlines vs. cuboids p-value (2-tailed repeated measure t-test): ' + str(human_lines_vs_cuboids_p_value),
            '\nlines vs. 0 p-value (2-tailed 1-sample t-test): ' + str(human_lines_vs_zero_p_value),
            '\ncuboids vs. 0 p-value (2-tailed 1-sample t-test): ' + str(human_cuboids_vs_zero_p_value),
            '\n\n-------- MODEL --------',
            '\nlines vs. cuboids p-value (2-tailed repeated measure t-test): ' + str(model_lines_vs_cuboids_p_value),
            '\nlines vs. 0 p-value (2-tailed 1-sample t-test): ' + str(model_lines_vs_zero_p_value),
            '\ncuboids vs. 0 p-value (2-tailed 1-sample t-test): ' + str(model_cuboids_vs_zero_p_value)]
    f_txt.writelines(text)


###########################
#    Prepare plotting:    #
###########################
errorbar_width = 4
markersize = 9
linewidth = 2
colormodel1 = 'sandybrown'
colormodel2 = 'cornflowerblue'
matplotlib.rcParams.update({'font.size': 18})

# Model results for lines:
results_model1 = 100 - np.mean(cleaned_results[:, 0, 1, :], 0) * 100
errors_model1 = np.std(cleaned_results[:, 0, 1, :], 0) * 100 / np.sqrt(np.size(cleaned_results, 0))

# Model results for cuboids:
results_model2 = 100 - np.mean(cleaned_results[:, 1, 1, :], 0) * 100
errors_model2 = np.std(cleaned_results[:, 1, 1, :], 0) * 100 / np.sqrt(np.size(cleaned_results, 0))


# Human results for lines:
results_humans1 = np.mean(human_results[:, 0, :], 0)
errors_humans1 = np.std(human_results[:, 0, :], 0) / np.sqrt(np.size(human_results, 0))

# Human results for cuboids:
results_humans2 = np.mean(human_results[:, 1, :], 0)
errors_humans2 = np.std(human_results[:, 1, :], 0) / np.sqrt(np.size(human_results, 0))

if plot_type=='improvement':
    results_model1 = results_model1[0] - results_model1
    results_model2 = results_model2[0] - results_model2
    results_humans1 = results_humans1[0] - results_humans1
    results_humans2 = results_humans2[0] - results_humans2


###########################
#        Plotting:        #
###########################
# Plot model performance
fig, ax = plt.subplots(figsize=(10,7))
x = np.arange(len(results_model1))
ax.errorbar(x, results_model1, yerr=errors_model1, marker='s', markersize=markersize, linewidth=linewidth, color=colormodel1, capsize=errorbar_width)
ax.errorbar(x, results_model2, yerr=errors_model2, marker='s', markersize=markersize, linewidth=linewidth, color=colormodel2, capsize=errorbar_width)

if plot_type=='improvement':
    ax.set_ylabel('Percent correct improvement [%point]')
elif plot_type=='threshold-like':
    ax.set_ylabel('Error rate [%]')
x_cat = ['1', '2', '3', '4', '5', '6', '7', '8']
ax.set_xticks(x)
ax.set_xticklabels(x_cat)
ax.set_xlabel('Routing iterations')
if save_plots:
    plt.savefig(parameters.logdir + 'final_performance_model.png')
else:
    plt.show()

# Plot human performance
fig, ax = plt.subplots(figsize=(10,7))
x = np.arange(len(results_humans1))
ax.errorbar(x, results_humans1, yerr=errors_humans1, marker='s', markersize=markersize, linewidth=linewidth, color=colormodel1, capsize=errorbar_width)
ax.errorbar(x, results_humans2, yerr=errors_humans2, marker='s', markersize=markersize, linewidth=linewidth, color=colormodel2, capsize=errorbar_width)

if plot_type=='improvement':
    ax.set_ylabel('Threshold improvement [arcsec]')
elif plot_type=='threshold-like':
    ax.set_ylabel('Threshold [arcsec]')
x_cat = ['1', '2', '3', '4', '5', '6', '7']
ax.set_xticks(x)
ax.set_xticklabels(x_cat)
ax.set_xlabel('Routing iterations')
if save_plots:
    plt.savefig(parameters.logdir + 'final_performance_humans.png')
else:
    plt.show()

print('... Finished script!')
print('-------------------------------')
