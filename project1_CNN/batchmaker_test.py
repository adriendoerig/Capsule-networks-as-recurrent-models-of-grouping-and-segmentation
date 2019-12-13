# -*- coding: utf-8 -*-
"""
Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets

This script can be used to explore the training and test set stimuli.
For more information about the individual parameters, have a look at
batchmaker.py

@author: Lynn Schmittwilken
"""

import numpy as np
import matplotlib.pyplot as plt
from parameters import parameters
from batchmaker import stim_maker_fn

imSize = parameters.im_size
shapeSize = parameters.shape_size
barWidth = parameters.bar_width
n_shapes = parameters.n_shapes
batch_size = 10
shape_types = parameters.shape_types
chosen_shape = 1
reduce_df = parameters.reduce_df
stim_idx = None
test = stim_maker_fn(imSize, shapeSize, barWidth)

# Check out how all stimuli look:
#test.plotAllStim([0, 1, 2, 3, 4, 5, 6])

# Check out a training batch:
[shape_1_images, shape_2_images, shapelabels_idx, vernierlabels_idx,
 nshapeslabels, nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = test.makeTrainBatch(
 shape_types, n_shapes, batch_size, reduce_df)
for i in range(batch_size):
    plt.imshow(np.squeeze(shape_1_images[i, :, :] + shape_2_images[i, :, :]))
    plt.pause(0.5)

# Check out a test batch:
#[vernier_images, shape_images,  shapelabels_idx, vernierlabels_idx,
# nshapeslabels, nshapeslabels_idx, x_vernier, y_vernier, x_shape, y_shape] = test.makeTestBatch(
# chosen_shape, n_shapes, batch_size, stim_idx, reduce_df)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(vernier_images[i, :, :] + shape_images[i, :, :]))
#    plt.pause(0.5)
