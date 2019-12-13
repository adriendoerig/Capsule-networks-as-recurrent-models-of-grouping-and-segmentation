# -*- coding: utf-8 -*-
"""
Experiment 2: The role of recurrent processing

Script to get the cleaned final results

@author: Lynn Schmittwilken
"""

import re
import numpy as np
import os
from parameters import parameters


print('Start cleaning process ...')

###########################
#      Preparations:      #
###########################
n_idx = parameters.n_idx
n_iterations = parameters.n_iterations
n_rounds = parameters.n_rounds
n_categories = len(parameters.test_crowding_data_paths)
routing_min = parameters.routing_min
routing_max = parameters.routing_max


# Getting final performance means:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Getting data:
            txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
                numbers = numbers[numbers!=4]
                numbers = numbers[numbers!=6]
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])
                
    # We want to clean the results from ceiled or floored performances:
    # 1. if the max performance for the flankers+vernier condition is smaller than
    #    0.55 (= floored)
    # 2. if the min performance for the flankers+vernier condition is smaller than
    #    0.45 (= floored and misleading)
    # 3. if the mean performance for the vernier-alone and flankers+vernier condition
    #    is larger than 0.95 (= ceiled)
    crit_value1 = np.max(np.squeeze(results[:, 1, :]), 0)
    crit_value2 = np.min(np.min(results[:, 0:2, :], 0), 0)
    crit_value3 = np.mean(np.mean(results[:, 0:2, :], 0), 0)

    crit_idx1 = np.where(crit_value1 < 0.55)
    crit_idx2 = np.where(crit_value2 < 0.45)
    crit_idx3 = np.where(crit_value3 > 0.95)
    
    crit_idx_all = np.unique(np.concatenate((crit_idx1, crit_idx2, crit_idx3), 1))
    good_idx_all = np.delete(np.arange(0, n_iterations), crit_idx_all)
    
    cleaned_results = results[:, :, good_idx_all]

    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_mean_cleaned_iterations_' + str(len(good_idx_all)) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.mean(cleaned_results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')


# Getting final performance stds:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Getting data:
            txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
                numbers = numbers[numbers!=4]
                numbers = numbers[numbers!=6]
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])
                
    # We can just reuse the indices from before:
    cleaned_results = results[:, :, good_idx_all]

    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_std_cleaned_iterations_' + str(len(good_idx_all)) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.std(cleaned_results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')

print('... Finished performance cleaning!')
print('-------------------------------------------------------')
