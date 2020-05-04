"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets

This script is creating and saving reconstruction of the input images
during TESTING.
The figures involve the input image, and the reconstructions of the top three
shape types that the network is most confident to be in the image

@author: Lynn Schmittwilken
"""

import logging
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from parameters import parameters
from batchmaker import stim_maker_fn
from capser_model_fn import model_fn


print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting get_reconstruction script...')
print('-------------------------------------------------------')


save_pckl = False

###########################
#    Helper functions:    #
###########################
def create_batch(chosen_shape, stim_idx, batch_size, parameters):
    '''
    Create a feed-dict with a batch of test stimuli as needed for the Estimator
    API. For more information about the Estimator API, look at:
    https://www.tensorflow.org/guide/estimators
    
    Parameters
    ----------
    chosen_shape: int
                  ShapeID of shape configuration that should be created
    stim_idx: int
              Choose the test condition that should be created
    batch_size: int
                Choose the desired batch size
    parameters: flags
                Contains all parameters defined in parameters.py
    
    Returns
    -------
    feed_dict
    '''
    im_size = parameters.im_size
    shape_size = parameters.shape_size
    bar_with = parameters.bar_width
    n_shapes = parameters.n_shapes
    centralize = parameters.centralized_shapes
    reduce_df = parameters.reduce_df
    test_noise = parameters.test_noise
    
    stim_maker = stim_maker_fn(im_size, shape_size, bar_with)
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels, nshapeslabels_idx, 
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(chosen_shape, n_shapes, batch_size, stim_idx, centralize, reduce_df)
    
    # Add some noise
    noise1 = np.random.uniform(test_noise[0], test_noise[1], [1])
    noise2 = np.random.uniform(test_noise[0], test_noise[1], [1])
    shape_1_images = shape_1_images + np.random.normal(0.0, noise1, [batch_size, im_size[0], im_size[1], parameters.im_depth])
    shape_2_images = shape_2_images + np.random.normal(0.0, noise2, [batch_size, im_size[0], im_size[1], parameters.im_depth])
    
    # Clip the pixel values
    shape_1_images = np.clip(shape_1_images, parameters.clip_values[0], parameters.clip_values[1])
    shape_2_images = np.clip(shape_2_images, parameters.clip_values[0], parameters.clip_values[1])
    
    feed_dict = {'shape_1_images': shape_1_images,
                 'shape_2_images': shape_2_images,
                 'shapelabels': shapelabels,
                 'nshapeslabels': nshapeslabels_idx,
                 'vernier_offsets': vernierlabels,
                 'x_shape_1': x_shape_1,
                 'y_shape_1': y_shape_1,
                 'x_shape_2': x_shape_2,
                 'y_shape_2': y_shape_2}
    return feed_dict

# Create an input function as needed for the Estimator API
def predict_input_fn(feed_dict):
    batch_size = feed_dict['shapelabels'].shape[0]
    
    shape_1_images = feed_dict['shape_1_images']
    shape_2_images = feed_dict['shape_2_images']
    shapelabels = feed_dict['shapelabels']
    nshapeslabels = feed_dict['nshapeslabels']
    vernier_offsets = feed_dict['vernier_offsets']
    x_shape_1 = feed_dict['x_shape_1']
    y_shape_1 = feed_dict['y_shape_1']
    x_shape_2 = feed_dict['x_shape_2']
    y_shape_2 = feed_dict['y_shape_2']
    
    dataset_test = tf.data.Dataset.from_tensor_slices((shape_1_images,
                                                       shape_2_images,
                                                       shapelabels,
                                                       nshapeslabels,
                                                       vernier_offsets,
                                                       x_shape_1,
                                                       y_shape_1,
                                                       x_shape_2,
                                                       y_shape_2))
    dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset_test = dataset_test.prefetch(2)
    
    # Create an iterator for the dataset_test and the above modifications.
    iterator = dataset_test.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = iterator.get_next()
    
    feed_dict = {'shape_1_images': shape_1_images,
                 'shape_2_images': shape_2_images,
                 'shapelabels': shapelabels,
                 'nshapeslabels': nshapeslabels,
                 'vernier_offsets': vernierlabels,
                 'x_shape_1': x_shape_1,
                 'y_shape_1': y_shape_1,
                 'x_shape_2': x_shape_2,
                 'y_shape_2': y_shape_2,
                 'mask_with_labels': False,
                 'is_training': False}

    return feed_dict


def plot_reconstructions(originals, results1, results2, results3, save=None):
    '''
    Plot the original input images in the first column, and the reconstructed
    images of the three shapes that have most activation in the next three
    columns.
    
    Parameters
    ----------
    originals: 3d array
               Selected number (batch_size) of input images that get passed to
               the network to be reconstructed
    results1, results2, results3: 3d arrays
                                  Reconstructed images of the three shapes that
                                  have most activation
    save: str
          if None do not save the figure, else provide a data path
    '''
    Nr = 4
    Nc = originals.shape[0]
    fig, axes = plt.subplots(Nc, Nr, figsize=(14,10))
    
    images = []
    for i in range(Nc):
        images.append(axes[i, 0].imshow(np.squeeze(originals[i,:,:,:])))
        axes[i, 0].axis('off')
        images.append(axes[i, 1].imshow(np.squeeze(results1[i,:,:,:])))
        axes[i, 1].axis('off')
        images.append(axes[i, 2].imshow(np.squeeze(results2[i,:,:,:])))
        axes[i, 2].axis('off')
        images.append(axes[i, 3].imshow(np.squeeze(results3[i,:,:,:])))
        axes[i, 3].axis('off')

    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        plt.close()


def save_images_in_file(originals, results1, results2, results3, save):
    # Save all the images for each configuration into a .pckl
    f = open(save, 'wb')
    obj = {'originals': originals,
           'results1': results1,
           'results2': results2,
           'results3': results3}
    pickle.dump(obj, f)
    f.close()


###########################
#       Main script:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(41)
tf.set_random_seed(41)

# Number of images to be reconstructed for each configuration
batch_size = 12

# How many networks have been trained?
n_iterations = parameters.n_iterations

# The total number of training steps is equal to n_steps*n_rounds
# (e.g. if n_steps=1000 and n_rounds=2, the network performance will be tested
#  using the test set after 1000 and 2000 steps)
n_rounds = parameters.n_rounds

# Number of test conditions used
n_idx = parameters.n_idx

# Total number of different shape types used
n_categories = len(parameters.test_shape_types)


for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'

    ##################################
    #     Testing / Predictions:     #
    ##################################
    # Lets have less logs:
    logging.getLogger().setLevel(logging.CRITICAL)        
    
    # The network should already be trained, so here we just do testing
    for n_category in range(n_categories):
        category_idx = parameters.test_shape_types[n_category]
        category = parameters.test_crowding_data_paths[n_category]
        print('-------------------------------------------------------')
        print('Reconstruct for ' + category)
        
        for stim_idx in range(n_idx):            
            # Get reconstructions using the following path:
            capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                            params={'log_dir': log_dir,
                                                    'get_reconstructions': True,
                                                    'batch_size': batch_size})
            feed_dict = create_batch(category_idx, stim_idx, batch_size, parameters)
            
            capser_out = list(capser.predict(lambda: predict_input_fn(feed_dict)))
            results1 = [p['decoder_output_img1'] for p in capser_out]
            results2 = [p['decoder_output_img2'] for p in capser_out]
            results3 = [p['decoder_output_img3'] for p in capser_out]
            
            # Plot and save the reconstruction images
            img_path = log_dir + '/reconstructions'
            img_file = img_path + '/' + category[21:] + str(stim_idx) + '.png'
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            originals = feed_dict['shape_1_images'] + feed_dict['shape_2_images']
            plot_reconstructions(originals, np.asarray(results1), np.asarray(results2),
                                 np.asarray(results3), img_file)
            
            # Also save reconstructions into pckl files for each configuration
            if save_pckl:
                img_pckl_path = img_path + '/pckls'
                if not os.path.exists(img_pckl_path):
                    os.mkdir(img_pckl_path)
                img_pckl_file = img_pckl_path + '/' + category[21:] + str(stim_idx) + '.pckl'
                save_images_in_file(originals, results1, results2, results3, img_pckl_file)


print('... Finished get_reconstruction script!')
print('-------------------------------------------------------')
