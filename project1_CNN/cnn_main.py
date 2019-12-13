# -*- coding: utf-8 -*-
"""
Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets

This is the main script to execute the training, evaluation and prediction of
the capsule network(s)

@author: Lynn Schmittwilken
"""

import logging
import numpy as np
import tensorflow as tf
import os
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
#from tensorflow.python.util import deprecation

from parameters import parameters
from cnn_model_fn import model_fn
from capser_input_fn import train_input_fn, eval_input_fn, predict_input_fn
from capser_functions import save_params, plot_results, plot_reconstruction, \
create_batch, predict_input_fn_2

###################################################

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('Chosen training procedure:', parameters.train_procedure)
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
get_reconstructions = True
reconstruction_batch_size = 4

# Decide how many networks you want to train and test
n_iterations = parameters.n_iterations

# Decide how many test conditions should be used
n_idx = 4

n_categories = len(parameters.test_crowding_data_paths)
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
    logging.getLogger().setLevel(logging.INFO)
    
    # Beholder to check on weights during training in tensorboard:
    beholder = Beholder(log_dir)
    beholder_hook = BeholderHook(log_dir)
    
    # Create the estimator:
    my_checkpointing_config = tf.estimator.RunConfig(keep_checkpoint_max = 2)  # Retain the 2 most recent checkpoints.
    
    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                    config=my_checkpointing_config,
                                    params={'log_dir': log_dir})
    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path),
                                      steps=parameters.eval_steps,
                                      throttle_secs=parameters.eval_throttle_secs)  

    ##################################
    #     Testing / Predictions:     #
    ##################################
    for idx_round in range(1, n_rounds+1):
        # For less annoying logs:
        logging.getLogger().setLevel(logging.CRITICAL)
        
        train_spec = tf.estimator.TrainSpec(train_input_fn,
                                            max_steps=parameters.n_steps*idx_round)
        tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)

        # Testing with crowding/uncrowding:
        cats = []
        res = []
        for n_category in range(n_categories):
            category_idx = parameters.test_shape_types[n_category]
            category = parameters.test_crowding_data_paths[n_category]
            cats.append(category[21:])
                
            print('-------------------------------------------------------')
            print('Compute vernier offset for ' + category)
            
            # Determine vernier_accuracy for our vernier/crowding/uncrowding stimuli
            results0 = np.zeros(shape=(n_idx,))
            for stim_idx in range(n_idx):
                test_filename = category + '/' + str(stim_idx) + '.tfrecords'
                
                
                ###################################
                #     Results: vernier acuity     #
                ###################################
                capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                params={'log_dir': log_dir,
                                                        'get_reconstructions': False})
                capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
                vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
                rank_pred_shapes = [p['rank_pred_shapes'] for p in capser_out]
                rank_pred_proba = [p['rank_pred_proba'] for p in capser_out]
    
                # Get the the results for averaging over several trained networks only from the final round:
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
                if get_reconstructions and idx_round==n_rounds:
                    # Lets also get some reconstructions for prediction mode using the following path:
                    capser = tf.estimator.Estimator(model_fn=model_fn,
                                                    model_dir=log_dir,
                                                    params={'log_dir': log_dir,
                                                            'get_reconstructions': True,
                                                            'batch_size': reconstruction_batch_size})
                    feed_dict = create_batch(category_idx, stim_idx, reconstruction_batch_size, parameters)
                    
                    capser_out = list(capser.predict(lambda: predict_input_fn_2(feed_dict)))
                    results1 = [p['decoder_output_img1'] for p in capser_out]
                    results2 = [p['decoder_output_img2'] for p in capser_out]
                    verniers = [p['decoder_vernier_img'] for p in capser_out]
                    
                    # Plotting and saving:
                    img_path = log_dir + '/uncrowding'
                    if not os.path.exists(img_path):
                        os.mkdir(img_path)
                    originals = feed_dict['shape_1_images'] + feed_dict['shape_2_images']
                    plot_reconstruction(originals, np.asarray(results1), np.asarray(results2), np.asarray(verniers), img_path + '/' + category[21:] + str(stim_idx) + '.png')


                # Saving ranking results:
                txt_ranking_file_name = log_dir + '/ranking_step_' + str(parameters.n_steps*idx_round) + '.txt'
                if not os.path.exists(txt_ranking_file_name):
                    with open(txt_ranking_file_name, 'w') as f:
                        f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
                else:
                    with open(txt_ranking_file_name, 'a') as f:
                        f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
            
            # Saving performance results:
            txt_file_name = log_dir + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
            if not os.path.exists(txt_file_name):
                with open(txt_file_name, 'w') as f:
                    f.write(category + ' : \t' + str(results0) + '\n')
            else:
                with open(txt_file_name, 'a') as f:
                    f.write(category + ' : \t' + str(results0) + '\n')

        # Plotting:
        plot_results(res, cats, n_idx, save=log_dir + '/uncrowding_results_step_' +
                     str(parameters.n_steps*idx_round) + '_noise_' +
                     str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.png')


# Saving final means considering all trained networks:
final_result_file = parameters.logdir + '/final_results_iterations_' + str(n_iterations) + '.txt'
final_results_mean = np.mean(results, 2)
for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]
    if not os.path.exists(final_result_file):
        with open(final_result_file, 'w') as f:
            f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')
    else:
        with open(final_result_file, 'a') as f:
            f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')

# Saving final stds considering all trained networks:
final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '.txt'
final_results_std = np.std(results, 2)
for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]
    if not os.path.exists(final_result_file):
        with open(final_result_file, 'w') as f:
            f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')
    else:
        with open(final_result_file, 'a') as f:
            f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')

print('... Finished capsnet script!')
print('-------------------------------------------------------')
