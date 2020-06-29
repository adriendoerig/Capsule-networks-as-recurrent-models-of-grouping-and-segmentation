"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 2: The role of recurrent processing

This script creates tfrecords files based on the stim_maker_fn class
(see batchmaker.py).
The tfrecords files are called in the input_fn of the Estimator API
(see capser_input_fn.py).

This code is inspired by the following youtube-video and code. Have a look if
you want to understand the details.
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

@author: Lynn Schmittwilken
"""

import sys
import os
import tensorflow as tf
import numpy as np
from parameters import parameters
from batchmaker import stim_maker_fn


##################################
#       Extra parameters:        #
##################################
training = 0
testing = 1
testing_crowding = 1

# Choose how many conditions should be included
n_idx = parameters.n_idx


##################################
#       Helper functions:        #
##################################
def wrap_int64(value):
    output = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return output

def wrap_bytes(value):
    output = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return output

def print_progress(count, total):
    percent_complete = float(count) / total
    msg = "\r- Progress: {0:.1%}".format(percent_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


##################################
#      tfrecords function:       #
##################################
def make_tfrecords(out_path, stim_maker, state, shape_types, n_samples,
                   train_procedure='random', stim_idx=None, reduce_df=False):
    '''
    Function to create tfrecord files based on stim_maker class
    
    Parameters
    ----------
    out_path: string
              Full data path including file name to which the dataset should be
              saved, e.g. 'datapath/filename.tfrecords'
    stim_maker: class
                Output of stim_maker_fn defined in batchmaker.py which defines
                the train and test stimuli
    state: string
           Tell the network whether it should create a training or test dataset
           by using either state='training' or state='testing'
    shape_types: list of ints
                 Pass a list with all shapeIDs corresponding to stim_maker class.
                 For training, it must be continuous from 0 to max
    train_procedure: string
                     Train procedure determines which and how many stimuli are
                     used for training. In this project, always use
                     train_procedure='random' in which case only shape_1 is
                     used for training and gets randomly selected from all
                     shape_types
    n_samples: int
               Full sample size of dataset
    stim_idx: int
              Based on the stim_idx, a test condition can be chosen. If
              stim_idx=None, a random condition is used. If stim_idx=0 the
              vernier-alone condition is chosen; if stim_idx=1 the crowding
              condition is chosen (=single flanker condition); if stim_idx=2 the
              uncrowding condition is chosen (multiple flankers condition either
              using flankers of one or two different types); if stim_idx=3 the
              control condition is chosen (no-crowding condition due to
              sufficient spacing between flankers and vernier)
    reduce_df: reduce_df: bool
               If reduce_df=False the stimulus group is placed randomly within
               the image. If reduce_df=True the stimulus group is still randomly
               placed within the image, however, the possibility of placement on
               the x-axis is controlled for the number of shape repetitions.
               Like this, it gets prevented that big stimulus groups are detected
               more easily just because their positioning on the x-axis is less
               variable
    '''
    
    print("\nConverting: " + out_path)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Create images one by one using stim_maker and save them
        for i in range(n_samples):
            print_progress(count=i, total=n_samples - 1)
            
            # Either create training or testing dataset
            if state=='training':
                [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
                 nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTrainBatch(
                 shape_types, 1, train_procedure, reduce_df)

            elif state=='testing':
                # The shape_types involve all configs in a dict (see parameters.py)
                test_configs = shape_types
                if len(test_configs) == 1:
                    # Either use the single test_config that is given:
                    config_idx = list(test_configs)[0]
                    chosen_config = test_configs[config_idx]
                else:
                    # Or chose one config randomly from all configs given for
                    # the validation set:
                    config_idx = np.random.randint(0, len(test_configs))
                    chosen_config = test_configs[str(config_idx)]

                [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
                 nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(
                 chosen_config, 1, stim_idx, reduce_df)

            # Convert the image to raw bytes.
            # Note: for this project, not all variables are needed
            shape_1_images_bytes = shape_1_images.tostring()
            shape_2_images_bytes = shape_2_images.tostring()
            shapelabels_bytes = shapelabels.tostring()
            nshapeslabels_bytes = nshapeslabels.tostring()
            nshapeslabels_idx_bytes = nshapeslabels_idx.tostring()
            vernierlabels_bytes = vernierlabels.tostring()
            x_shape_1_bytes = x_shape_1.tostring()
            y_shape_1_bytes = y_shape_1.tostring()
            x_shape_2_bytes = x_shape_2.tostring()
            y_shape_2_bytes = y_shape_2.tostring()

            # Create a dict with the data to save in the TFRecords file
            data = {'shape_1_images': wrap_bytes(shape_1_images_bytes),
                    'shape_2_images': wrap_bytes(shape_2_images_bytes),
                    'shapelabels': wrap_bytes(shapelabels_bytes),
                    'nshapeslabels': wrap_bytes(nshapeslabels_bytes),
                    'nshapeslabels_idx': wrap_bytes(nshapeslabels_idx_bytes),
                    'vernierlabels': wrap_bytes(vernierlabels_bytes),
                    'x_shape_1': wrap_bytes(x_shape_1_bytes),
                    'y_shape_1': wrap_bytes(y_shape_1_bytes),
                    'x_shape_2': wrap_bytes(x_shape_2_bytes),
                    'y_shape_2': wrap_bytes(y_shape_2_bytes)}

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
    return


###################################
#     Create tfrecords files:     #
###################################
print('\n-------------------------------------------------------')
print('Creating tfrecords files of type:', parameters.train_procedure)

stim_maker = stim_maker_fn(parameters.im_size, parameters.shape_size, parameters.bar_width, parameters.offset)

if not os.path.exists(parameters.data_path):
    os.mkdir(parameters.data_path)


# Create the training set:
if training:
    mode = 'training'
    shape_types_train = parameters.shape_types
    train_procedure = 'random'
    make_tfrecords(parameters.train_data_path, stim_maker, mode, shape_types_train,
                   parameters.n_train_samples, train_procedure, reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of training set')
    print('-------------------------------------------------------')


# Create the validation and the test set that uses the same stimuli as in
# the training set:
if testing:
    mode = 'training'
    shape_types_train = parameters.shape_types
    train_procedure = 'random'

    # Validation set:
    make_tfrecords(parameters.val_data_path, stim_maker, mode, shape_types_train,
                   parameters.n_test_samples, train_procedure, reduce_df=parameters.reduce_df)

    # Individual test sets:
    for i in range(len(parameters.test_data_paths)):
        # We use +1 here to skip a vernier-vernier configuration
        chosen_shape = shape_types_train[i+1]
        test_file_path = parameters.test_data_paths[i]
        make_tfrecords(test_file_path, stim_maker, mode, chosen_shape,
                       parameters.n_test_samples, train_procedure, reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of regular validation and test sets')
    print('-------------------------------------------------------')


# Create the validation and the test set that uses vernier-alone and flankers
# test stimuli:
if testing_crowding:
    mode = 'testing'
    test_configs = parameters.test_configs[0]
    
    # Validation sets:
    make_tfrecords(parameters.val_crowding_data_path, stim_maker, mode, test_configs,
                   parameters.n_test_samples,  reduce_df=parameters.reduce_df)

    # Individual test sets:
    for i in range(len(test_configs)):
        test_config = {str(i): test_configs[str(i)]}
        test_data_path = parameters.test_crowding_data_paths[i]
        if not os.path.exists(test_data_path):
            os.mkdir(test_data_path)
        for stim_idx in range(n_idx):
            test_file_path = test_data_path + '/' + str(stim_idx) + '.tfrecords'
            make_tfrecords(test_file_path, stim_maker, mode, test_config,
                           parameters.n_test_samples, stim_idx=stim_idx, 
                           reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of crowding validaton and test sets')
    print('-------------------------------------------------------')
