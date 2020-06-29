"""
Capsule Networks as Recurrent Models of Grouping and Segmentation
Experiment 2: The role of recurrent processing
This script defines all stimuli of the training and test datasets.
It involves all basic shapes used in the paper (verniers, lines, cuboids,
shuffled cuboids) and an additional category which was not used (rectancles).
@author: Lynn Schmittwilken
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


##################################
#      stim_maker class fn:      #
##################################
class stim_maker_fn:
    def __init__(self, imSize, shapeSize, barWidth, offset=1, transparent_cuboids=False):
        self.imSize = imSize

        # Extra parameters for 3d nature of cuboids
        shapeDepth = shapeSize[2]
        depthW = int(np.floor((np.tan(45 / 180 * np.pi) * shapeDepth)))
        depthH = int(np.floor((np.sin(45 / 180 * np.pi) * shapeDepth)))
        self.depthW = depthW
        self.depthH = depthH
        self.transparent_cuboids = transparent_cuboids

        # All shape patches are supposed to have the same height
        self.patchHeight = shapeSize[1] + depthH
        self.shapeWidth = shapeSize[0]
        self.shapeHeight = shapeSize[1]
        self.barWidth = barWidth         # Line width, for small images only use =1
        self.vernierOffsetHeight = 1     # Vernier offset vertically (fixed)
        self.offset = offset             # offset used between all shapes (0 or 1)

        # Additional fixed parameters for the experiment 2:
        # Fixed number of shapes in the input image (except lines):
        self.shape_repetitions = 2

        # Shapes must face one another (relevant for cuboids):
        self.face_each_other = 1

        # Lines are repeated more often, to make sure that the network does not
        # learn too basic differences between the shape types (e.g. number of pixels):
        if self.transparent_cuboids:
            self.line_repetitions = [2, 4, 6, 8]
        else:
            self.line_repetitions = [2, 4, 6]

        # Fix maximal offset between the stimuli
        self.max_offset_line = self.offset * 6
        self.max_offset_stim = self.offset * 6

        if not np.mod(shapeSize[1] + self.vernierOffsetHeight, 2) == 0:
            raise SystemExit('\nshapeHeight + vernierOffsetHeight has to be even!')

    def drawVernier(self, offset, offset_direction):
        '''
        Draw a vernier stimulus within a patch of size [patchHeight, vernier width].

        Parameters
        ----------
        offset: int
                vernier offset
        offset_direction: int
                          offset_direction: 0=r, 1=l

        Returns
        -------
        patch: 2d array
               image patch of size [patchHeight, vernier width] including a
               vernier stimulus
        '''
        patchHeight = self.patchHeight
        height = self.shapeHeight
        depthH = self.depthH
        barW = self.barWidth
        offsetW = offset
        offsetH = self.vernierOffsetHeight

        vernierSize = int((height - offsetH) / 2)
        patch = np.zeros([patchHeight, 2 * barW + offsetW], dtype=np.float32)
        patch[depthH:depthH + vernierSize, 0:barW] = 1
        patch[depthH + offsetH + vernierSize:depthH + offsetH + vernierSize * 2, barW + offsetW:] = 1

        if offset_direction:
            patch = np.fliplr(patch)
        return patch

    def drawLines(self, offset=0):
        '''
        Draw a line within a patch of size [patchHeight, line width + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a line stimulus
        '''
        patchHeight = self.patchHeight
        height = self.shapeHeight
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, barW + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, offset:offset + barW] = 1
        return patch

    def drawRectangles(self, offset=0):
        '''
        Draw a rectangle within a patch of size [patchHeight, rectangle width
        + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a rectangle
        '''
        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, width + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, offset + width - barW:offset + width] = 1
        patch[depthH:depthH + height, offset:offset + barW] = 1
        patch[depthH:depthH + barW, offset:offset + width] = 1
        patch[depthH + height - barW:depthH + height, offset:offset + width] = 1
        return patch

    def drawCuboidsR(self, offset=0):
        '''
        Draw a cuboid facing right within a patch of size [patchHeight, cuboid
        width + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a cuboid facing right
        '''

        # To make sure that the drawing function is not out of borders
        adjust = 1

        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthW = self.depthW
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, width + depthW + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, depthW + offset + width - barW:depthW + offset + width] = 1
        patch[depthH:depthH + height, depthW + offset:depthW + offset + barW] = 1
        patch[depthH:depthH + barW, depthW + offset:depthW + offset + width] = 1
        patch[depthH + height - barW:depthH + height, depthW + offset:depthW + offset + width] = 1

        patch[0:barW, offset:offset + width] = 1
        patch[0:height, offset:offset + barW] = 1

        row1, col1 = draw.line(0, offset, depthH, offset + depthW)
        row2, col2 = draw.line(height - adjust, offset, height + depthH - adjust, offset + depthW)
        row3, col3 = draw.line(0, width + offset - adjust, depthH, width + depthW + offset - adjust)
        
        patch[row1, col1] = 1
        patch[row2, col2] = 1
        patch[row3, col3] = 1
        
        if self.transparent_cuboids:
            patch[height - adjust:height + barW - adjust, offset:offset + width] = 1
            patch[0:height, offset + width:offset + width + barW] = 1
            row4, col4 = draw.line(height - adjust, width + offset - adjust, height + depthH - adjust,
                                   width + depthW + offset - adjust)
            patch[row4, col4] = 1
            
        return patch

    def drawCuboidsL(self, offset):
        '''
        Draw a cuboid facing left within a patch of size [patchHeight, cuboid
        width + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a cuboid facing left
        '''
        patch = self.drawCuboidsR(offset)
        patch = np.fliplr(patch)
        return patch

    def drawShuffledCuboidsR(self, offset=0):
        '''
        Draw a shuffled cuboid facing right within a patch of size
        [patchHeight, cuboid width + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a shuffled cuboid facing right
        '''
        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthW = self.depthW
        depthH = self.depthH
        barW = self.barWidth
        patchWidth = width + depthW + 2 * offset

        patch = np.zeros([patchHeight, patchWidth], dtype=np.float32)

        # The line close to the vernier should always stay the same:
        patch[depthH:depthH + height, depthW + offset + width - barW:depthW + offset + width] = 1

        # All others should be random
        rnd1 = np.random.randint(0, patchHeight - height)
        rnd2 = np.random.randint(offset, patchWidth - offset - barW)
        patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - height)
        rnd2 = np.random.randint(offset, patchWidth - offset - barW)
        patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row1, col1 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row2, col2 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row3, col3 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        patch[row1, col1] = 1
        patch[row2, col2] = 1
        patch[row3, col3] = 1
        
        if self.transparent_cuboids:
            rnd1 = np.random.randint(0, patchHeight - barW)
            rnd2 = np.random.randint(offset, patchWidth - offset - width)
            patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1
            
            rnd1 = np.random.randint(0, patchHeight - height)
            rnd2 = np.random.randint(offset, patchWidth - offset - barW)
            patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1
            
            rnd1 = np.random.randint(0, patchHeight - depthH)
            rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
            row4, col4 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)
            patch[row4, col4] = 1
        
        return patch

    def drawShuffledCuboidsL(self, offset=0):
        '''
        Draw a shuffled cuboid facing left within a patch of size
        [patchHeight, cuboid width + offset*2].

        Parameters
        ----------
        offset: int
                offset added to patch width (default: 0)

        Returns
        -------
        patch: 2d array
               image patch including a shuffled cuboid facing left
        '''
        patch = self.drawShuffledCuboidsR(offset)
        patch = np.fliplr(patch)
        return patch

    def drawShape(self, shapeID, offset, offset_direction=0):
        '''
        Draw a chosen shape.
        For this, it should be defined here how each shape looks like.
        Importantly, the shapeID needs to range from 0 to the selected number of
        different shapes.

        Parameters
        ----------
        shapeID: int
                 shapeID of the shape that should be drawn
        offset: int
                vernier offset / offset added to patch width for other shapes
        offset_direction: int
                          if the chosen shape is a vernier, you can choose the
                          offset direction (0=r, 1=l)

        Returns
        -------
        patch: 2d array
               patch including the chosen shape with varying widths depending
               on chosen shape and a height of patchHeight
        '''
        if shapeID == 0:
            patch = self.drawVernier(offset, offset_direction)
        if shapeID == 1:
            patch = self.drawLines(offset)
        if shapeID == 6:
            patch = self.drawRectangles(offset)
        if shapeID == 2:
            patch = self.drawCuboidsR(offset)
        if shapeID == 3:
            patch = self.drawShuffledCuboidsR(offset)
        if shapeID == 4:
            patch = self.drawCuboidsL(offset)
        if shapeID == 5:
            patch = self.drawShuffledCuboidsL(offset)
        return patch

    def plotAllStim(self, shape_types, offset):
        '''
        Function to visualize the chosen shape_types in a single plot.

        Parameters
        ----------
        shape_types: list of ints
                     choose all the shapeIDs, you want to compare within the plot
                     (e.g [0, 1, 4])
        offset: int
                vernier offset / offset added to patch width for other shapes
        '''
        row = 0
        col = 0
        full_width = 0

        # Due to varying widths, we need to get the total width here
        for i in range(len(shape_types)):
            ID = shape_types[i]
            patch = self.drawShape(ID, offset)
            full_width += np.size(patch, 1)

        image = np.zeros([self.patchHeight, full_width], dtype=np.float32)

        for i in range(len(shape_types)):
            ID = shape_types[i]
            patch = self.drawShape(ID, offset)
            tmp_width = np.size(patch, 1)
            image[row:row+self.patchHeight, col:col+tmp_width] += patch
            col += tmp_width
        plt.figure()
        plt.imshow(image)
        return

    def makeTestBatch(self, crowding_config, batch_size, stim_idx=None,
                      reduce_df=False):
        '''
        Create one batch of the test dataset for the condition chosen with stim_idx.

        Parameters
        ----------
        crowding_config: list
                         Within this list, define the configuration for testing
                         (e.g. [1, 0, 1] for line, vernier, line)
        batch_size: int
                    chosen batch size
        stim_idx: int
                  Based on the stim_idx, a condition can be chosen. If stim_idx=None,
                  a random condition is used. If stim_idx=0 the vernier-alone
                  condition is chosen; if stim_idx=1 the vernier and the flankers
                  are presented together; if stim_idx=2 the flankers-alone
                  condition is chosen
        reduce_df: bool
                   If reduce_df=False the stimulus group is placed randomly within
                   the image. If reduce_df=True the stimulus group is still randomly
                   placed within the image, however, the possibility of placement on
                   the x-axis is controlled for the number of shape repetitions.
                   Like this, it gets prevented that big stimulus groups are detected
                   more easily just because their positioning on the x-axis is less
                   variable

        Returns
        -------
        vernier_images: 4d array
                        Array that involves a full batch of gray-scale images
                        only including the vernier stimulus
        shape_images: 4d array
                      Array that involves full batch of gray-scale images only
                      including the flanker stimuli
        shapelabels_idx: 2d array
                         Array that involves full batch of shapeIDs for the vernier
                         and flanker stimulus
        vernierlabels_idx: 1d vector
                           Vector that involves full batch of vernier offset
                           directions
        Other returns: Diverse
                       Not relevant in this project but needed for code flow
        '''

        imSize = self.imSize
        patchHeight = self.patchHeight
        selected_shape = crowding_config[0]
        offset = self.offset
        shape_repetitions = self.shape_repetitions

        vernier_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shape_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            vernier_image = np.zeros(imSize, dtype=np.float32)
            shape_image = np.zeros(imSize, dtype=np.float32)
            row = np.random.randint(0, imSize[0] - patchHeight)

            if stim_idx is None:
                idx = np.random.randint(0, 2)
            else:
                idx = stim_idx

            # For the reduction of the dfs, we need the patch widths:
            # In this case, the cuboid patch width is the largest
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(0, offset, offset_direction)
            vernierpatch_width = np.size(vernier_patch, 1)
            maxWidth = 2 * (self.shapeWidth + self.depthW + offset * 2) + vernierpatch_width

            # Define different test conditions:
            if idx == 0:
                # Vernier-alone test stimuli:
                selected_repetitions = 1
                nshapes_label = 0
                totalWidth = vernierpatch_width

                if reduce_df:
                    # We want to make the degrees of freedom for position on
                    # the x axis fair.
                    # For this condition, we have to reduce the image size
                    # depending on the actual patch width
                    imSize_adapted = imSize[1] - maxWidth + totalWidth
                    imStart = int((imSize[1] - imSize_adapted) / 2)
                    col = np.random.randint(imStart, imStart + imSize_adapted - totalWidth + 1)

                else:
                    col = np.random.randint(0, imSize[1] - totalWidth)

                vernier_image[row:row + patchHeight, col:col + vernierpatch_width] += vernier_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx == 1:
                # Vernier-Flanker test stimuli
                selected_repetitions = shape_repetitions
                nshapes_label = 1

                totalWidth = 0
                for i in range(len(crowding_config)):
                    shape = crowding_config[i]
                    shape_patch = self.drawShape(shape, offset)
                    totalWidth += np.size(shape_patch, 1)

                if reduce_df:
                    # We want to make the degrees of freedom for position on
                    # the x axis fair.
                    # For this condition, we have to reduce the image size
                    # depending on the actual patch width
                    imSize_adapted = imSize[1] - maxWidth + totalWidth
                    imStart = int((imSize[1] - imSize_adapted) / 2)
                    col = np.random.randint(imStart, imStart + imSize_adapted - totalWidth + 1)

                else:
                    col = np.random.randint(0, imSize[1] - totalWidth)

                # Take the shape coordinates:
                x_shape_ind, y_shape_ind = col, row

                # Loop through the configuration for the flanker stimuli
                for i in range(len(crowding_config)):
                    shape = crowding_config[i]
                    shape_patch = self.drawShape(shape, offset)
                    patchWidth = np.size(shape_patch, 1)
                    if shape == 0:
                        vernier_image[row:row + patchHeight, col:col + patchWidth] += vernier_patch
                        x_vernier_ind, y_vernier_ind = col, row
                    else:
                        shape_image[row:row + patchHeight, col:col + patchWidth] += shape_patch
                    col += patchWidth

            vernier_images[idx_batch, :, :] = vernier_image
            shape_images[idx_batch, :, :] = shape_image
            shapelabels_idx[idx_batch, 0] = 0
            shapelabels_idx[idx_batch, 1] = selected_shape
            nshapeslabels[idx_batch] = selected_repetitions
            nshapeslabels_idx[idx_batch] = nshapes_label
            vernierlabels_idx[idx_batch] = offset_direction
            x_vernier[idx_batch] = x_vernier_ind
            y_vernier[idx_batch] = y_vernier_ind
            x_shape[idx_batch] = x_shape_ind
            y_shape[idx_batch] = y_shape_ind

        # add the color channel for tensorflow:
        vernier_images = np.expand_dims(vernier_images, -1)
        shape_images = np.expand_dims(shape_images, -1)
        return [vernier_images, shape_images, shapelabels_idx, vernierlabels_idx,
                nshapeslabels, nshapeslabels_idx, x_vernier, y_vernier, x_shape, y_shape]


    def makeTrainBatch(self, shape_types, batch_size, train_procedure='random',
                       reduce_df=False):
        '''
        Create one batch of the training dataset with each two groups of
        shape_type

        Parameters
        ----------
        shape_types: list or int
                     Either list including all possible shapeIDs of which one ID
                     is randomly chosen to create the stimulus, or just a single
                     shapeID that is used to create the stimulus.
        batch_size: int
                    chosen batch size
        train_procedure: string
                     Train procedure determines which and how many stimuli are
                     used for training. In this project, always use
                     train_procedure='random' in which case only shape_1 is
                     used for training and gets randomly selected from all
                     shape_types
        reduce_df: bool
                   If reduce_df=False the stimulus group is placed randomly within
                   the image. If reduce_df=True the stimulus group is still randomly
                   placed within the image, however, the possibility of placement on
                   the x-axis is controlled for the number of shape repetitions.
                   Like this, it gets prevented that big stimulus groups are detected
                   more easily just because their positioning on the x-axis is less
                   variable

        Returns
        -------
        shape_1_images: 4d array
                        Array that involves full batch of gray-scale images
                        including the flanker stimuli in 50% of the cases and
                        otherwise including a single vernier stimulus
        shape_2_images: 4d array
                        The script was adopted from experiment 1. However, shape_2
                        will not be used in the following anymore
        shapelabels_idx: 2d array
                         Array that involves full batch of shapeIDs for the vernier
                         and flanker stimulus
        vernierlabels_idx: 1d vector
                           Vector that involves full batch of vernier offset
                           directions
        Other returns: Diverse
                       Not relevant in this project but needed for code flow
        '''

        imSize = self.imSize
        patchHeight = self.patchHeight
        shape_repetitions = self.shape_repetitions
        face_each_other = self.face_each_other
        line_reps = self.line_repetitions

        # Set the max random offset between the stimuli:
        max_offset_line = self.max_offset_line
        max_offset_stim = self.max_offset_stim

        # The offset is set to zero, since we introduce a rd_offset afterwards
        offset = 0
        maxPatchWidth = self.shapeWidth + self.depthW + offset * 2

        shape_1_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shape_2_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        x_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            shape_1_image = np.zeros(imSize, dtype=np.float32)
            shape_2_image = np.zeros(imSize, dtype=np.float32)

            try:
                # Every second image should contain a vernier:
                if np.random.rand(1) < 0.5:
                    selected_shape_1 = 0
                else:
                    selected_shape_1 = np.random.randint(0, len(shape_types))
                selected_shape_2 = np.random.randint(1, len(shape_types))
            except:
                # if only one shape is passed, just use this shape_type
                selected_shape_1 = 0
                selected_shape_2 = shape_types

            # Create shape images:
            if selected_shape_1 == 0:
                # If the first shape is a vernier, only repeat it once and use
                # offset_direction 0=r or 1=l
                idx_n_shapes_1 = 0
                selected_repetitions_1 = 1
                rd_offset = 0
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, self.offset, offset_direction)
            elif selected_shape_1 == 1:
                # The line stimulus is the only one that can be repeated more often:
                idx_n_shapes_1 = np.random.randint(0, len(line_reps)) + 1  # atm, idx_n_shapes_1=1 means 2 reps
                selected_repetitions_1 = line_reps[np.random.randint(0, len(line_reps))]
                rd_offset = np.random.randint(self.offset, max_offset_line + 1)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, offset)
            else:
                # In all other cases, repeat shape random times but at least
                # once and set offset_direction to 2=no vernier
                idx_n_shapes_1 = 1
                selected_repetitions_1 = shape_repetitions
                rd_offset = np.random.randint(self.offset, max_offset_stim + 1)
                #                rd_offset = np.random.randint(self.offset, imSize[1]-maxPatchWidth*shape_repetitions)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, offset)

            # This script was adopted from experiment 1. However, shape_2
            # will not be used in the following anymore
            idx_n_shapes_2 = 0
            selected_repetitions_2 = 1
            shape_2_patch = self.drawShape(0, offset)
            row_shape_2 = 0
            col_shape_2_init = 0
            col_shape_2 = col_shape_2_init

            # For the reduction of the dfs, we need to know the patch widths:
            # In this case, the cuboid patch width is biggest
            shape1patch_width = np.size(shape_1_patch, 1)
            shape2patch_width = np.size(shape_2_patch, 1)

            row_shape_1 = np.random.randint(0, imSize[0] - patchHeight)
            if reduce_df:
                # We want to make the degrees of freedom for position on the x axis fair.
                # For this condition, we have to reduce the image size depending on
                # the actual patch width
                if idx_n_shapes_1 == 0:
                    imSize_adapted = imSize[1] - maxPatchWidth * shape_repetitions + shape1patch_width * selected_repetitions_1 - 1
                else:
                    imSize_adapted = imSize[1] - maxPatchWidth * shape_repetitions + shape1patch_width * selected_repetitions_1
                imStart = int((imSize[1] - imSize_adapted) / 2)
                col_shape_1_init = np.random.randint(imStart, imStart + imSize_adapted - shape1patch_width * selected_repetitions_1 - rd_offset)
                col_shape_1 = col_shape_1_init

            else:
                col_shape_1_init = np.random.randint(0, imSize[1] - shape1patch_width * selected_repetitions_1 - rd_offset)
                col_shape_1 = col_shape_1_init

            if selected_shape_1 == 0:
                # If there is a vernier, only use one shape per image:
                shape_1_image[row_shape_1:row_shape_1 + patchHeight,
                col_shape_1:col_shape_1 + shape1patch_width] += shape_1_patch
            else:
                # Repeat shape_1 selected_repetitions times if not vernier:
                for i in range(selected_repetitions_1):
                    shape_1_image[row_shape_1:row_shape_1 + patchHeight,
                    col_shape_1:col_shape_1 + shape1patch_width] += shape_1_patch
                    col_shape_1 += shape1patch_width + rd_offset
                    if face_each_other == 1:
                        shape_1_patch = np.fliplr(shape_1_patch)

            # Repeat shape_2 selected_repetitions times:
            for i in range(selected_repetitions_2):
                shape_2_image[row_shape_2:row_shape_2 + patchHeight,
                col_shape_2:col_shape_2 + shape2patch_width] += shape_2_patch
                col_shape_2 += shape1patch_width + rd_offset

            shape_1_images[idx_batch, :, :] = shape_1_image
            shape_2_images[idx_batch, :, :] = shape_2_image
            shapelabels_idx[idx_batch, 0] = selected_shape_1
            shapelabels_idx[idx_batch, 1] = selected_shape_2
            vernierlabels_idx[idx_batch] = offset_direction
            nshapeslabels[idx_batch, 0] = selected_repetitions_1
            nshapeslabels[idx_batch, 1] = selected_repetitions_2
            nshapeslabels_idx[idx_batch, 0] = idx_n_shapes_1
            nshapeslabels_idx[idx_batch, 1] = idx_n_shapes_2
            x_shape_1[idx_batch] = col_shape_1_init
            y_shape_1[idx_batch] = row_shape_1
            x_shape_2[idx_batch] = col_shape_2_init
            y_shape_2[idx_batch] = row_shape_2

        # add the color channel for tensorflow:
        shape_1_images = np.expand_dims(shape_1_images, -1)
        shape_2_images = np.expand_dims(shape_2_images, -1)
        return [shape_1_images, shape_2_images, shapelabels_idx, vernierlabels_idx,
                nshapeslabels, nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2]
