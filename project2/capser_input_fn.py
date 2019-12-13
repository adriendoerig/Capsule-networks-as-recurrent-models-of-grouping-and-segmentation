# -*- coding: utf-8 -*-
"""
Experiment 2: The role of recurrent processing

This input function is used for the Estimator API of tensorflow
(see capser_main.py).
Also, data augmentation is realized within this script.
For more information about the Estimator API, look at:
https://www.tensorflow.org/guide/estimators

@author: Lynn Schmittwilken
"""

import tensorflow as tf
from parameters import parameters


########################################
#     Parse tfrecords training set:    #
########################################
def parse_tfrecords_train(serialized_data):
    with tf.name_scope('Parsing_trainset'):
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = {'shape_1_images': tf.FixedLenFeature([], tf.string),
                    'shape_2_images': tf.FixedLenFeature([], tf.string),
                    'shapelabels': tf.FixedLenFeature([], tf.string),
                    'vernierlabels': tf.FixedLenFeature([], tf.string)}
    
        # Parse the serialized data so we get a dict with our data.
        parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    
        # Get the images as raw bytes and decode afterwards.
        shape_1_images = parsed_data['shape_1_images']
        shape_1_images = tf.decode_raw(shape_1_images, tf.float32)
        shape_1_images = tf.cast(shape_1_images, tf.float32)
    
        shape_2_images = parsed_data['shape_2_images']
        shape_2_images = tf.decode_raw(shape_2_images, tf.float32)
        shape_2_images = tf.cast(shape_2_images, tf.float32)
        
        # Get the labels associated with the image and decode.
        shapelabels = parsed_data['shapelabels']
        shapelabels = tf.decode_raw(shapelabels, tf.float32)
        shapelabels = tf.cast(shapelabels, tf.int64)
        
        vernierlabels = parsed_data['vernierlabels']
        vernierlabels = tf.decode_raw(vernierlabels, tf.float32)

        # Reshaping:
        shape_1_images = tf.reshape(shape_1_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shape_2_images = tf.reshape(shape_2_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shapelabels = tf.reshape(shapelabels, [2])
        vernierlabels = tf.reshape(vernierlabels, [1])
        
        # Make sure this image is empty (only used it for experiment 1)
        shape_2_images = tf.zeros([parameters.im_size[0], parameters.im_size[1], parameters.im_depth], tf.float32)


    ##################################
    #       Data augmentation:       #
    ##################################
    with tf.name_scope('Data_augmentation_trainset'):
        # Add some random gaussian TRAINING noise (always):
        noise1 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
        noise2 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
        shape_1_images = tf.add(shape_1_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0, stddev=noise1))
        shape_2_images = tf.add(shape_2_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0, stddev=noise2))
    
    
        # Adjust brightness and contrast by a random factor
        def bright_contrast():
            shape_1_images_augmented = tf.image.random_brightness(shape_1_images,
                                                                  parameters.delta_brightness)
            shape_2_images_augmented = tf.image.random_brightness(shape_2_images,
                                                                  parameters.delta_brightness)
            shape_1_images_augmented = tf.image.random_contrast(
                    shape_1_images_augmented,parameters.delta_contrast[0],
                    parameters.delta_contrast[1])
            shape_2_images_augmented = tf.image.random_contrast(
                    shape_2_images_augmented, parameters.delta_contrast[0],
                    parameters.delta_contrast[1])
            return shape_1_images_augmented, shape_2_images_augmented
        
        def contrast_bright():
            shape_1_images_augmented = tf.image.random_contrast(
                    shape_1_images, parameters.delta_contrast[0], parameters.delta_contrast[1])
            shape_2_images_augmented = tf.image.random_contrast(
                    shape_2_images, parameters.delta_contrast[0], parameters.delta_contrast[1])
            shape_1_images_augmented = tf.image.random_brightness(shape_1_images_augmented,
                                                                  parameters.delta_brightness)
            shape_2_images_augmented = tf.image.random_brightness(shape_2_images_augmented,
                                                                  parameters.delta_brightness)
            return shape_1_images_augmented, shape_2_images_augmented
    
        # Maybe change contrast and brightness:
        if parameters.allow_contrast_augmentation:
            pred = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1.,
                                             dtype=tf.float32), 0.5)
            shape_1_images, shape_2_images = tf.cond(pred, bright_contrast,
                                                     contrast_bright)
    
        # Clip the pixel values
        shape_1_images = tf.clip_by_value(shape_1_images, parameters.clip_values[0],
                                          parameters.clip_values[1])
        shape_2_images = tf.clip_by_value(shape_2_images, parameters.clip_values[0],
                                          parameters.clip_values[1])

    return [shape_1_images, shape_2_images, shapelabels, vernierlabels]


########################################
#      Parse tfrecords test set:       #
########################################
def parse_tfrecords_test(serialized_data):
    with tf.name_scope('Parsing_testset'):
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = {'shape_1_images': tf.FixedLenFeature([], tf.string),
                    'shape_2_images': tf.FixedLenFeature([], tf.string),
                    'shapelabels': tf.FixedLenFeature([], tf.string),
                    'vernierlabels': tf.FixedLenFeature([], tf.string)}
    
        # Parse the serialized data so we get a dict with our data.
        parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    
        # Get the images as raw bytes and decode afterwards.
        shape_1_images = parsed_data['shape_1_images']
        shape_1_images = tf.decode_raw(shape_1_images, tf.float32)
        shape_1_images = tf.cast(shape_1_images, tf.float32)
    
        shape_2_images = parsed_data['shape_2_images']
        shape_2_images = tf.decode_raw(shape_2_images, tf.float32)
        shape_2_images = tf.cast(shape_2_images, tf.float32)
        
        # Get the labels associated with the image and decode.
        shapelabels = parsed_data['shapelabels']
        shapelabels = tf.decode_raw(shapelabels, tf.float32)
        shapelabels = tf.cast(shapelabels, tf.int64)
        
        vernierlabels = parsed_data['vernierlabels']
        vernierlabels = tf.decode_raw(vernierlabels, tf.float32)
        vernierlabels = tf.cast(vernierlabels, tf.int64)
    
        # Reshaping:
        shape_1_images = tf.reshape(shape_1_images, [parameters.im_size[0],parameters.im_size[1], parameters.im_depth])
        shape_2_images = tf.reshape(shape_2_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shapelabels = tf.reshape(shapelabels, [2])
        vernierlabels = tf.reshape(vernierlabels, [1])
    
        # For the test and validation set, we dont really need data augmentation,
        # but we'd still like some TEST noise
        noise1 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1],
                                   tf.float32)
        noise2 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1],
                                   tf.float32)
        shape_1_images = tf.add(shape_1_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                mean=0.0, stddev=noise1))
        shape_2_images = tf.add(shape_2_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                mean=0.0, stddev=noise2))
        
        # Clip the pixel values
        shape_1_images = tf.clip_by_value(shape_1_images, parameters.clip_values[0],
                                          parameters.clip_values[1])
        shape_2_images = tf.clip_by_value(shape_2_images, parameters.clip_values[0],
                                          parameters.clip_values[1])
    
    return [shape_1_images, shape_2_images, shapelabels, vernierlabels]


###########################
#     Input function:     #
###########################
def input_fn(filenames, stage, parameters, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)
    
    # Depending on whether we use the train or test set, different parsing functions
    # are used:
    if stage=='train' or stage=='eval':
        dataset = dataset.map(parse_tfrecords_train, num_parallel_calls=64)
        
        # Read a buffer of the given size and randomly shuffle it:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Allow for infinite reading of data
        num_repeat = parameters.n_epochs

    else:
        dataset = dataset.map(parse_tfrecords_test, num_parallel_calls=64)
        
        # Don't shuffle the data and only go through it once:
        num_repeat = 1
        
    # Repeat the dataset the given number of times and get a batch of data
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(parameters.batch_size, drop_remainder=True)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    [shape_1_images, shape_2_images, shapelabels, vernierlabels] = iterator.get_next()

    if stage=='train':
        feed_dict = {'shape_1_images': shape_1_images,
                     'shape_2_images': shape_2_images,
                     'shapelabels': shapelabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': True,
                     'is_training': True}

    else:
        feed_dict = {'shape_1_images': shape_1_images,
                     'shape_2_images': shape_2_images,
                     'shapelabels': shapelabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': False,
                     'is_training': False}
    return feed_dict, shapelabels


##############################
#   Final input functions:   #
##############################
def train_input_fn():
    return input_fn(filenames=parameters.train_data_path, stage='train', parameters=parameters)

def eval_input_fn(filename):
    return input_fn(filenames=filename, stage='eval', parameters=parameters)

def predict_input_fn(filenames):
    return input_fn(filenames=filenames, stage='test', parameters=parameters)
