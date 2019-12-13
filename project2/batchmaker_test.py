# -*- coding: utf-8 -*-
"""
Experiment 2: The role of recurrent processing

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
offset = parameters.offset
batch_size = 10
shape_types = parameters.shape_types
crowding_config = [2, 0, 4]
reduce_df = parameters.reduce_df
stim_idx = 2
test = stim_maker_fn(imSize, shapeSize, barWidth, offset)

# Check out how all stimuli look:
#test.plotAllStim([0, 1, 2, 3, 4, 5], offset)

# Check out a training batch:
#[shape_1_images, shape_2_images, shapelabels_idx, vernierlabels_idx] = test.makeTrainBatch(
#shape_types, batch_size, reduce_df)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(shape_1_images[i, :, :]))
#    plt.pause(0.5)

# Check out a test batch:
[vernier_images, shape_images,  shapelabels_idx, vernierlabels_idx] = test.makeTestBatch(
crowding_config, batch_size, stim_idx, reduce_df)
for i in range(batch_size):
    plt.imshow(np.squeeze(vernier_images[i, :, :] + shape_images[i, :, :]))
    plt.pause(0.5)
