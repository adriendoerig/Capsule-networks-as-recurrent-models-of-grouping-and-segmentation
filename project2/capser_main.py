"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 2: The role of recurrent processing

This is the main script to execute the training, evaluation and prediction of
the capsule network(s)

@author: Lynn Schmittwilken
"""

import re
import logging
import numpy as np
import tensorflow as tf
import os

# Optional: use Beholder to check on weights during training in tensorboard
#from tensorboard.plugins.beholder import Beholder
#from tensorboard.plugins.beholder import BeholderHook

from parameters import parameters
from capser_model_fn import model_fn
from capser_input_fn import train_input_fn, eval_input_fn, predict_input_fn
from capser_functions import save_params, plot_uncrowding_results


print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')


###########################
#      Preparations:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# Decide how many networks you want to train and test
n_iterations = parameters.n_iterations

# With n_rounds, you can control the number of evaluations using the test set
# The total number of training steps is equal to n_steps*n_rounds
# (e.g. if n_steps=1000 and n_rounds=2, the network performance will be tested
#  using the test set after 1000 and 2000 steps)
n_rounds = parameters.n_rounds

# Decide how many test conditions should be used
n_idx = parameters.n_idx

# We will test the network performance with an increasing the number of 
# routing iterations:
routing_min = parameters.routing_min
routing_max = parameters.routing_max

# Save parameters from parameter file for reproducibility
if not os.path.exists(parameters.logdir):
    os.mkdir(parameters.logdir)
save_params(parameters.logdir, parameters)

# Total number of different shape types used
n_categories = len(parameters.test_crowding_data_paths)

# Initialize results
results = np.zeros(shape=(n_categories, n_idx, n_iterations))


###########################
#       Main script:      #
###########################
for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'

    ##################################
    #    Training and evaluation:    #
    ##################################
    # Output the loss in the terminal every few steps:
    logging.getLogger().setLevel(logging.INFO)

    # Optional: use Beholder to check on weights during training in tensorboard
#    beholder = Beholder(log_dir)
#    beholder_hook = BeholderHook(log_dir)

    # Create the estimator (Retain the 2 most recent checkpoints)
    checkpointing_config = tf.estimator.RunConfig(keep_checkpoint_max = 2)

    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                    config=checkpointing_config,
                                    params={'log_dir': log_dir,
                                            'iter_routing': parameters.train_iter_routing})
    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path),
                                      steps=parameters.eval_steps,
                                      throttle_secs=parameters.eval_throttle_secs)


    for idx_round in range(1, n_rounds+1):
        # Train for n_steps*n_rounds but testing after each round
        train_spec = tf.estimator.TrainSpec(train_input_fn,
                                            max_steps=parameters.n_steps*idx_round)
        tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)

        ##################################
        #     Testing / Predictions:     #
        ##################################
        # Lets have less logs:
        logging.getLogger().setLevel(logging.CRITICAL)

        # Testing for each chosen routing iteration (between routing_min and
        # routing_max) and each test stimulus category:
        for idx_routing in range(routing_min, routing_max+1):
            log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'
            if not os.path.exists(log_dir_results):
                os.mkdir(log_dir_results)

            cats = []
            res = []
            for n_category in range(n_categories):
                category = parameters.test_crowding_data_paths[n_category]
                cats.append(category[21:])

                print('-------------------------------------------------------')
                print('Compute vernier offset for ' + category)

                results0 = np.zeros(shape=(n_idx,))
                for stim_idx in range(n_idx):
                    # Load relevant tfrecord test stimulus file
                    test_filename = category + '/' + str(stim_idx) + '.tfrecords'

                    # Lets get all the network performance results we need:
                    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                    params={'log_dir': log_dir,
                                                            'iter_routing': idx_routing})
                    capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
                    vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
                    rank_pred_shapes = [p['rank_pred_shapes'] for p in capser_out]
                    rank_pred_proba = [p['rank_pred_proba'] for p in capser_out]

                    # Get all the results per round:
                    results0[stim_idx] = np.mean(vernier_accuracy)
                    results1 = np.unique(rank_pred_shapes)
                    results2 = np.mean(rank_pred_proba, 0)
                    res.append(np.mean(vernier_accuracy))

                    print('Finished calculations for stimulus type ' + str(stim_idx))
                    print('Result: ' + str(results0[stim_idx]) + '; test_samples used: ' + str(len(vernier_accuracy)))


                    # Save ranking results (which shapes did the network recognize
                    # with the highest probabilities?):
                    txt_ranking_file_name = log_dir_results + '/ranking_step_' + str(parameters.n_steps*idx_round) + '.txt'
                    if not os.path.exists(txt_ranking_file_name):
                        with open(txt_ranking_file_name, 'w') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
                    else:
                        with open(txt_ranking_file_name, 'a') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')

                # Save network performance results:
                txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) + \
                '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
                if not os.path.exists(txt_file_name):
                    with open(txt_file_name, 'w') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')
                else:
                    with open(txt_file_name, 'a') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')

            # Plot and save crowding/uncrowding results:
            plot_uncrowding_results(res, cats, n_idx,
                                    save=log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) +
                                    '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.png')


###########################
#    Final performance    #
###########################
# Get final performance means:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Get relevant data:
            txt_file_name = (log_dir_results + '/uncrowding_results_step_' +
                             str(parameters.n_steps*n_rounds) + '_noise_' +
                             str(parameters.test_noise[0]) + '_' +
                             str(parameters.test_noise[1]) + '.txt')

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


    # Save final means:
    final_result_file = (parameters.logdir + '/final_results_mean_iterations_' +
                         str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt')
    final_results = np.mean(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')


# Get final performance stds:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Get relevant data:
            txt_file_name = (log_dir_results + '/uncrowding_results_step_' +
                             str(parameters.n_steps*n_rounds) + '_noise_' +
                             str(parameters.test_noise[0]) + '_' +
                             str(parameters.test_noise[1]) + '.txt')

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


    # Save final stds:
    final_result_file = (parameters.logdir + '/final_results_std_iterations_' +
                         str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt')
    final_results = np.std(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')


print('... Finished capsnet script!')
print('-------------------------------------------------------')
