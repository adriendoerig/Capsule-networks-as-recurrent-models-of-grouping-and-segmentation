# -*- coding: utf-8 -*-
"""
Experiment 2: The role of recurrent processing

This is the main script to execute the training, evaluation and prediction of
the capsule network(s)

@author: Lynn Schmittwilken
"""


import re
#import logging
import numpy as np
import tensorflow as tf
import os
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
#from tensorflow.python.util import deprecation

from parameters import parameters
from capser_model_fn import model_fn
from capser_input_fn import train_input_fn, eval_input_fn, predict_input_fn
from capser_functions import save_params, plot_results, plot_reconstruction, \
create_batch, predict_input_fn_2

###################################################

# Disable annoying logs:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.logging.set_verbosity(tf.logging.ERROR)

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('-------------------------------------------------------')

# Turn off most deprecation logs:
#deprecation._PRINT_DEPRECATION_WARNINGS = False


###########################
#      Preparations:      #
###########################
# For reproducibility (seed for weights not affected):
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# Decide whether you want to save reconstructions during testing
get_reconstructions = False
reconstruction_batch_size = 4

# Decide how many networks you want to train and test
n_iterations = parameters.n_iterations

# Decide how many test conditions should be used
n_idx = parameters.n_idx
n_categories = len(parameters.test_crowding_data_paths)

# We will increase the number of routing iterations:
routing_min = parameters.routing_min
routing_max = parameters.routing_max
results = np.zeros(shape=(n_categories, n_idx, n_iterations))

# With n_rounds, you can control the number of evaluations
# The number of full training steps is equal to n_steps*n_rounds
n_rounds = parameters.n_rounds

# Save parameters from parameter file for reproducibility
if not os.path.exists(parameters.logdir):
    os.mkdir(parameters.logdir)
save_params(parameters.logdir, parameters)


###########################
#       Main script:      #
###########################
if get_reconstructions and not parameters.decode_reconstruction:
    # Make sure that you trained the decoder if you want to have reconstructions
    raise SystemExit('\nPROBLEM: You asked for reconstructions but the reconstruction decoder was not trained! (see parameters.py: decode_reconstruction)')


for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'
    
    ##################################
    #    Training and evaluation:    #
    ##################################
    # Output the loss in the terminal every few steps:
#    logging.getLogger().setLevel(logging.INFO)
    
    # Beholder to check on weights during training in tensorboard:
    beholder = Beholder(log_dir)
    beholder_hook = BeholderHook(log_dir)
    
    # Create the estimator:
    my_checkpointing_config = tf.estimator.RunConfig(keep_checkpoint_max = 2)  # Retain the 2 most recent checkpoints.
    
    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                    config=my_checkpointing_config,
                                    params={'log_dir': log_dir,
                                            'iter_routing': parameters.train_iter_routing})
    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path),
                                      steps=parameters.eval_steps,
                                      throttle_secs=parameters.eval_throttle_secs)
    
    ##################################
    #     Testing / Predictions:     #
    ##################################
    for idx_round in range(1, n_rounds+1):
        # Lets have less annoying logs:
#        logging.getLogger().setLevel(logging.CRITICAL)
        
        train_spec = tf.estimator.TrainSpec(train_input_fn,
                                            max_steps=parameters.n_steps*idx_round)
        tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)

        for idx_routing in range(routing_min, routing_max+1):
            log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'
            if not os.path.exists(log_dir_results):
                os.mkdir(log_dir_results)

            # Testing with crowding/uncrowding:
            cats = []
            res = []
            for n_category in range(n_categories):
                category = parameters.test_crowding_data_paths[n_category]
                cats.append(category[21:])
                
                # For the reconstructions:
                test_configs = parameters.test_configs[0]
                category_idx = {str(n_category): test_configs[str(n_category)]}

                print('-------------------------------------------------------')
                print('Compute vernier offset for ' + category)

                # Determine vernier_accuracy for our vernier/crowding/uncrowding stimuli
                results0 = np.zeros(shape=(n_idx,))
                for stim_idx in range(n_idx):
                    test_filename = category + '/' + str(stim_idx) + '.tfrecords'

                    ###################################
                    #          Performance            #
                    ###################################
                    # Lets get all the results we need:
                    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                    params={'log_dir': log_dir,
                                                            'get_reconstructions': False,
                                                            'iter_routing': idx_routing})
                    capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
                    vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
                    rank_pred_shapes = [p['rank_pred_shapes'] for p in capser_out]
                    rank_pred_proba = [p['rank_pred_proba'] for p in capser_out]

                    # Get the the results for averaging over several trained
                    # networks only from the final round:
                    if idx_round==n_rounds:
                        results[n_category, stim_idx, idx_execution] = np.mean(vernier_accuracy)

                    # Get all the other results per round:
                    results0[stim_idx] = np.mean(vernier_accuracy)
                    results1 = np.unique(rank_pred_shapes)
                    results2 = np.mean(rank_pred_proba, 0)
                    res.append(np.mean(vernier_accuracy))

                    print('Finished calculations for stimulus type ' + str(stim_idx))
                    print('Result: ' + str(results0[stim_idx]) + '; test_samples used: ' + str(len(vernier_accuracy)))
                    
                    
                    ###################################
                    #         Reconstructions         #
                    ###################################
                    if get_reconstructions:
                        capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                        params={'log_dir': log_dir,
                                                                'get_reconstructions': True,
                                                                'batch_size': reconstruction_batch_size,
                                                                'iter_routing': idx_routing})
                        feed_dict = create_batch(category_idx, stim_idx, reconstruction_batch_size, parameters)
                        
                        capser_out = list(capser.predict(lambda: predict_input_fn_2(feed_dict)))
                        rec_results1 = [p['decoder_output_img1'] for p in capser_out]
                        rec_results2 = [p['decoder_output_img2'] for p in capser_out]
                        rec_verniers = [p['decoder_vernier_img'] for p in capser_out]
                        
                        # Plotting and saving:
                        img_path = log_dir_results + '/uncrowding'
                        if not os.path.exists(img_path):
                            os.mkdir(img_path)
                        originals = feed_dict['shape_1_images'] + feed_dict['shape_2_images']
                        plot_reconstruction(originals, np.asarray(rec_results1), np.asarray(rec_results2), np.asarray(rec_verniers),
                                     img_path + '/' + category[21:] + str(stim_idx) + '.png')

                    

                    # Saving ranking results:
                    txt_ranking_file_name = log_dir_results + '/ranking_step_' + str(parameters.n_steps*idx_round) + '.txt'
                    if not os.path.exists(txt_ranking_file_name):
                        with open(txt_ranking_file_name, 'w') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
                    else:
                        with open(txt_ranking_file_name, 'a') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')

                # Saving performance results:
                txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) + \
                '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
                if not os.path.exists(txt_file_name):
                    with open(txt_file_name, 'w') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')
                else:
                    with open(txt_file_name, 'a') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')

            # Plotting:
            plot_results(res, cats, n_idx, save=log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) +
                                                           '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.png')




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


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_mean_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.mean(results, 2)
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


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
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


# Get reconstructions:
#os.system('python my_capser_get_reconstructions.py')