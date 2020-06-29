# -*- coding: utf-8 -*-
"""
Experiment 2: The role of recurrent processing

Exemplary parameters file used to get the data presented in our paper
@author: Lynn Schmittwilken
"""

import tensorflow as tf
import numpy as np

flags = tf.app.flags


###########################
#          Paths          #
###########################
# General data paths for logs:
data_path = './data'
MODEL_NAME = '_logs'
flags.DEFINE_string('data_path', data_path, 'path where all data files are located')
flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')

# Training data set involving all shape defined in shape_types:
flags.DEFINE_string('train_data_path', data_path+'/train.tfrecords', 'path for tfrecords with training set')

# Validation data set involving all shape defined in shape_types:
flags.DEFINE_string('val_data_path', data_path+'/val.tfrecords', 'path for tfrecords with validation set')

# Test data set involving all shape defined in shape_types except the first (=vernier).
# Note that these paths have to correspond to the shapes defined in shape_types
# but skipping the first (e.g. [0, 1, 2] corresponding to verniers, lines and
# cuboids as defined in batchmaker.py should correspond to
# [data_path + 'lines.tfrecords', data_path + 'cuboids.tfrecords'])
flags.DEFINE_list('test_data_paths',
                  [data_path+'/test_lines.tfrecords',
#                   data_path+'/test_rectangles.tfrecords',
                   data_path+'/test_cuboids.tfrecords',
                   data_path+'/test_shuffled_cuboids.tfrecords'
                   ], 'path for tfrecords with test set')

# Validation data set involving all shape defined in test_shape_types:
flags.DEFINE_string('val_crowding_data_path', data_path+'/val_crowding.tfrecords', 'path for tfrecords with validation crowding set')

# Test data set involving all configurations defined in test_shape_types.
# Note that these paths have to correspond to the configs defined in test_shape_types
# (e.g. [[1, 0, 1], [2, 0, 4] corresponding to line-vernier-line and
# cuboidR-vernier-cuboidL as defined in batchmaker.py should correspond to
# [data_path + 'lines', data_path + 'cuboids'])
flags.DEFINE_list('test_crowding_data_paths',
                  [data_path+'/test_crowding_lines',
#                   data_path+'/test_crowding_rectangles',
                   data_path+'/test_crowding_cuboids',
                   data_path+'/test_crowding_shuffled_cuboids'
                   ], 'path for tfrecords with test crowding set')


###########################
#     Reproducibility     #
###########################
flags.DEFINE_integer('random_seed', None, 'if not None, set seed for weights initialization')


###########################
#   Stimulus parameters   #
###########################
# IMPORTANT:
    # After changing any of the following stimulus parameters, you need to
    # recreate the datasets
flags.DEFINE_string('train_procedure', 'random', 'only random is possible')
flags.DEFINE_boolean('reduce_df', True,  'if true, the degrees of freedom for position on the x axis get adapted')
flags.DEFINE_integer('n_train_samples', 240000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 2400, 'number of samples in the test set')
flags.DEFINE_integer('n_idx', 2, 'number of test conditions')

im_size = [16, 48]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_list('shape_size', [14, 11, 6], 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')
flags.DEFINE_integer('offset', 1, 'offset between shapes and vernier offset width')
flags.DEFINE_boolean('transparent_cuboids', 'False', 'use opaque or transparent cuboids')

# Define which shapes should be used during training and testing.
# For this, have a look at the corresponding shapes in batchmaker.py
# IMPORTANT:
    # 1. The numbers in shape_types always have to range from 0 to max.
    # 2. test_data_paths have to match the shapes defined in shape_types skipping
    # the first
    # 3. test_crowding_data_paths have to match the shapes defined in test_shape_types
shape_types = [0, 1, 2, 3]
test_configs = {'0': [1, 0, 1],
                '1': [2, 0, 4],
                '2': [3, 0, 5]}
flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('test_configs', [test_configs], 'pool of shapes (see batchmaker)')


###########################
#    Data augmentation    #
###########################
flags.DEFINE_list('train_noise', [0.0, 0.02], 'amount of added random Gaussian noise')
flags.DEFINE_list('test_noise', [0.1, 0.12], 'amount of added random Gaussian noise')
flags.DEFINE_list('clip_values', [0., 1.], 'min and max pixel value for every image')
flags.DEFINE_boolean('allow_contrast_augmentation', True, 'augment by changing contrast and brightness')
flags.DEFINE_float('delta_brightness', 0.1, 'factor to adjust brightness (+/-), must be non-negative')
flags.DEFINE_list('delta_contrast', [0.9, 1.1], 'min and max factor to adjust contrast, must be non-negative')


###########################
#   Network parameters    #
###########################
# Conv and primary caps parameters:
caps1_nmaps = len(shape_types)
caps1_ndims = 1

kernel1 = 4
kernel2 = 4
kernel3 = 5
stride1 = 1
stride2 = 1
stride3 = 2

# Calculate the output dimensions of the last convolutional layer in order to
# calculate the total number of primary capsules:
dim1 = int(np.round((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3))
dim2 = int(np.round((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3))

conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1, 'padding': 'valid'}
conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2, 'padding': 'valid'}
conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3, 'padding': 'valid'}
flags.DEFINE_list('conv_params', [conv1_params, conv2_params, conv3_params], 'list with the conv parameters')


flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')


# Output caps:
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 3, 'second caps layer, number of dims')


# Decoder reconstruction:
flags.DEFINE_string('rec_decoder_type', 'conv', 'use fc or conv layers for decoding (only with 3 conv layers)')
flags.DEFINE_integer('n_hidden_reconstruction_1', 512, 'size of hidden layer 1 in decoder')
flags.DEFINE_integer('n_hidden_reconstruction_2', 1024, 'size of hidden layer 2 in decoder')
flags.DEFINE_integer('n_output', im_size[0]*im_size[1], 'output size of the decoder')


###########################
#    Hyperparameters      #
###########################
# For training
flags.DEFINE_integer('batch_size', 48, 'batch size')
flags.DEFINE_float('learning_rate', 0.0004, 'chosen learning rate for training')
flags.DEFINE_float('learning_rate_decay_steps', 500, 'decay for cosine decay restart')

flags.DEFINE_integer('n_epochs', None, 'number of epochs, if None allow for indifinite readings')
flags.DEFINE_integer('n_steps', 4500, 'number of steps')
flags.DEFINE_integer('n_rounds', 1, 'number of evaluations; full training steps is equal to n_steps times this number')
flags.DEFINE_integer('n_iterations', 50, 'number of trained networks')

flags.DEFINE_integer('buffer_size', 1024, 'buffer size')
flags.DEFINE_integer('eval_steps', 50, 'frequency for eval spec; u need at least eval_steps*batch_size stimuli in the validation set')
flags.DEFINE_integer('eval_throttle_secs', 150, 'minimal seconds between evaluation passes')
flags.DEFINE_integer('train_iter_routing', 8, 'number of iterations in routing algorithm during training')
flags.DEFINE_integer('routing_min', 1, 'min number of iterations in routing algorithm during testing')
flags.DEFINE_integer('routing_max', 8, 'max number of iterations in routing algorithm during testing')
flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')


###########################
#         Losses          #
###########################
flags.DEFINE_boolean('decode_reconstruction', True, 'decode the reconstruction and use reconstruction loss')

flags.DEFINE_boolean('decode_nshapes', False, 'decode the number of shapes and use nshapes loss')
nshapes_loss = 'xentropy'
flags.DEFINE_string('nshapes_loss', nshapes_loss, 'currently either xentropy or squared_diff')

flags.DEFINE_boolean('decode_location', False, 'decode the shapes locations and use location loss')
location_loss = 'xentropy'
flags.DEFINE_string('location_loss', location_loss, 'currently either xentropy or squared_diff')


# Control magnitude of losses
flags.DEFINE_float('alpha_vernieroffset', 1., 'alpha for vernieroffset loss')
flags.DEFINE_float('alpha_margin', 0.5, 'alpha for margin loss')
flags.DEFINE_float('alpha_shape_1_reconstruction', 0.0005, 'alpha for reconstruction loss for vernier image (reduce_sum)')
flags.DEFINE_float('alpha_shape_2_reconstruction', 0.0001, 'alpha for reconstruction loss for shape image (reduce_sum)')

if nshapes_loss=='xentropy':
    flags.DEFINE_float('alpha_nshapes', 0.4, 'alpha for nshapes loss')
elif nshapes_loss=='squared_diff':
    flags.DEFINE_float('alpha_nshapes', 0.002, 'alpha for nshapes loss')

if location_loss=='xentropy':
    flags.DEFINE_float('alpha_x_shape_1_loss', 0.1, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shape_1_loss', 0.1, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_shape_2_loss', 0.1, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_shape_2_loss', 0.1, 'alpha for loss of y coordinate of vernier')
elif location_loss=='squared_diff':
    flags.DEFINE_float('alpha_x_shape_1_loss', 0.000004, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shape_1_loss', 0.00005, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_shape_2_loss', 0.000004, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_shape_2_loss', 0.00005, 'alpha for loss of y coordinate of vernier')


# Margin loss extras
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')


###########################
#     Regularization       #
###########################

flags.DEFINE_boolean('dropout', False, 'use dropout after conv layers 1&2')
flags.DEFINE_boolean('batch_norm_conv', False, 'use batch normalization between every conv layer')
flags.DEFINE_boolean('batch_norm_reconstruction', False, 'use batch normalization for the reconstruction decoder layers')
flags.DEFINE_boolean('batch_norm_vernieroffset', False, 'use batch normalization for the vernieroffset loss layer')
flags.DEFINE_boolean('batch_norm_nshapes', False, 'use batch normalization for the nshapes loss layer')
flags.DEFINE_boolean('batch_norm_location', False, 'use batch normalization for the location loss layer')


parameters = tf.app.flags.FLAGS
