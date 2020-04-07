"""
Capsule Networks as Recurrent Models of Grouping and Segmentation

Experiment 1: Crowding and Uncrowding Naturally Occur in CapsNets

This script defines all stimuli of the training and test datasets.
It involves all basic shapes used in the paper (verniers, squares, circles,
polygons, stars) and an additional category which can be used to prevent
overfitting (stuff-category)

@author: Lynn Schmittwilken
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


##################################
#      stim_maker class fn:      #
##################################
class stim_maker_fn:
    def __init__(self, imSize, shapeSize, barWidth):
        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth

    
    def drawVernier(self, offset_direction, zoom=0):
        '''
        Draw a vernier stimulus within a patch of size [shapeSize, shapeSize].
        
        Parameters
        ----------
        offset_direction: int
                          offset_direction: 0=r, 1=l
        zoom: int
              neg/pos number to de-/increase shape size
        
        Returns
        -------
        fullPatch: 2d array
                   image patch of size [shapeSize, shapeSize] including a
                   vernier stimulus
        '''
        barHeight = int((self.shapeSize+zoom)/4 - (self.barWidth)/4)
        offsetHeight = 1
        
        # We chose the minimum distance between verniers to be one pixel
        if barHeight/2 < 2:
            offset_size = 1
        else:
            offset_size = np.random.randint(1, barHeight/2)
        patch = np.zeros((2*barHeight+offsetHeight, 2*self.barWidth+offset_size), dtype=np.float32)
        patch[0:barHeight, 0:self.barWidth] = 1
        patch[barHeight+offsetHeight:, self.barWidth+offset_size:] = 1
        
        if offset_direction:
            patch = np.fliplr(patch)
        fullPatch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch
        return fullPatch


    def drawSquare(self, zoom=0):
        '''
        Draw a square within a patch of size [shapeSize, shapeSize].
        
        Parameters
        ----------
        zoom: int
              neg/pos number to de-/increase shape size.
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including a square
        '''
        zoom = np.abs(zoom)
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        patch[zoom:self.barWidth+zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, zoom:self.barWidth+zoom] = 1
        patch[self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom] = 1
        return patch


    def drawCircle(self, zoom=0, eps=1):
        '''
        Draw a circle within a patch of size [shapeSize, shapeSize].
        
        Parameters
        ----------
        zoom: int
              neg/pos number to de-/increase shape size.
        eps: int
             needed to control potential empty spots in a shape
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including a circle
        '''
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        radius = (self.shapeSize+zoom)/2
        t = np.linspace(0, np.pi*2, self.shapeSize*4)
        for i in range(1,self.barWidth*eps+1):
            row = np.floor((radius-i/eps) * np.cos(t)+radius - zoom/2)
            col = np.floor((radius-i/eps) * np.sin(t)+radius - zoom/2)
            patch[row.astype(np.int), col.astype(np.int)] = 1
        return patch

    
    def drawPolygon(self, nSides, phi, zoom=0, eps=1):
        '''
        Draw a polygon within a patch of size [shapeSize, shapeSize].
        
        Parameters
        ----------
        nSides: int
                number of sides (e.g. 4 for diamond, 6 for hexagon)
        phi: float
             angle for rotation
        zoom: int
              neg/pos number to de-/increase shape size.
        eps: int
             needed to control potential empty spots in a shape
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including a polygon
        '''
        
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        # If shapeSize is very small, you might want to move the shape a little
        # to perfectly fit the vernier stimulus inside. This can be controlled
        # with slide_factor
        slide_factor = -1
        radius = (self.shapeSize+zoom)/2
        t = np.linspace(0+phi, np.pi*2+phi, nSides+1)
        for i in range(1,self.barWidth*eps+1):
            rowCorner = (np.round((radius-i/eps) * np.sin(t)+radius - zoom/2 + slide_factor))
            colCorner = (np.round((radius-i/eps) * np.cos(t)+radius - zoom/2 + slide_factor))
            for n in range(len(rowCorner)-1):
                rowLines, colLines = draw.line(rowCorner.astype(np.int)[n], colCorner.astype(np.int)[n],
                                               rowCorner.astype(np.int)[n+1], colCorner.astype(np.int)[n+1])
                patch[rowLines, colLines] = 1
        return patch

    
    def drawStar(self, nSides, phi1, phi2, depth, zoom=1, eps=1):
        '''
        Draw a star within a patch of size [shapeSize, shapeSize].
        
        Parameters
        ----------
        nSides: int
                number of sides (e.g. 4 for diamond, 6 for hexagon)
        phi1: float
              angle for rotation of outer corners
        phi2: float
              angle for rotation of inner corners
        depth: float
               control the distance between inner and outer corners
        zoom: int
              neg/pos number to de-/increase shape size.
        eps: int
             needed to control potential empty spots in a shape
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including a star
        '''
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        # If shapeSize is very small, you might want to move the shape a little
        # to perfectly fit the vernier stimulus inside. This can be controlled
        # with slide_factor
        slide_factor = -1
        radius_big = (self.shapeSize + zoom)/2
        radius_small = self.shapeSize/depth
        tExt = np.linspace(0+phi1, np.pi*2+phi1, nSides+1)
        tInt = np.linspace(0+phi2, np.pi*2+phi2, nSides+1)
        for i in range(1,self.barWidth*eps+1):
            rowCornerExt = (np.round((radius_big-i/eps) * np.sin(tExt)+radius_big - zoom/2 + slide_factor))
            colCornerExt = (np.round((radius_big-i/eps) * np.cos(tExt)+radius_big - zoom/2 + slide_factor))
            rowCornerInt = (np.round((radius_small-i/eps) * np.sin(tInt)+radius_big - zoom/2 + slide_factor))
            colCornerInt = (np.round((radius_small-i/eps) * np.cos(tInt)+radius_big - zoom/2 + slide_factor))
            for n in range(0, len(rowCornerExt)-1, 1):
                rowLines, colLines = draw.line(rowCornerExt.astype(np.int)[n], colCornerExt.astype(np.int)[n],
                                               rowCornerInt.astype(np.int)[n], colCornerInt.astype(np.int)[n])
                patch[rowLines, colLines] = 1
                rowLines, colLines = draw.line(rowCornerExt.astype(np.int)[n+1], colCornerExt.astype(np.int)[n+1],
                                               rowCornerInt.astype(np.int)[n], colCornerInt.astype(np.int)[n])
                patch[rowLines, colLines] = 1
        return patch
    
    
    def drawStuff(self, n_lines):
        '''
        Draw some random lines within a patch of size [shapeSize, shapeSize].
        This stimulus can be used to prevent overfitting.
        
        Parameters
        ----------
        nLines: int
                number of randomly drawn lines
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including random lines.
        '''
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)

        for n in range(n_lines):
            (r1, c1, r2, c2) = np.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = 1
        return patch

    
    def drawShape(self, shapeID, offset_direction=0):
        '''
        Draw a chosen shape.
        For this, it should be defined here how each shape looks like.
        Importantly, the shapeID needs to range from 0 to the selected number of
        different shapes.
        
        Parameters
        ----------
        shapeID: int
                 shapeID of the shape that should be drawn
        offset_direction: int
                          if the chosen shape is a vernier, you can choose the
                          offset direction (0=r, 1=l)
        
        Returns
        -------
        fullPatch: 2d array
                   patch of size [shapeSize, shapeSize] including the chosen shape
        '''
        if shapeID == 0:
            patch = self.drawVernier(offset_direction, -2)
        if shapeID == 1:
            patch = self.drawSquare(-1)
        if shapeID == 2:
            patch = self.drawCircle(-1)
        if shapeID == 3:
            patch = self.drawPolygon(4, 0)
        if shapeID == 4:
            patch = self.drawStar(4, np.pi/4, np.pi/2, 3., 5)
        if shapeID == 5:
            patch = self.drawPolygon(6, 0)
        if shapeID == 6:
            patch = self.drawStar(6, 0, np.pi/6, 3, 0)
        if shapeID == 7:
            patch = self.drawStuff(5)
        return patch

    
    def plotAllStim(self, shape_types):
        '''
        Function to visualize the chosen shape_types in a single plot.
        
        Parameters
        ----------
        shape_types: list of ints
                     choose all the shapeIDs, you want to compare within the plot
                     (e.g [0, 1, 4]).
        '''

        image = np.zeros([self.shapeSize, len(shape_types)*(self.shapeSize)], dtype=np.float32)
        row = 0
        col = 0
        for i in range(len(shape_types)):
            ID = shape_types[i]
            patch = self.drawShape(shapeID=ID)
            image[row:row+self.shapeSize, col:col+self.shapeSize] += patch
            col += self.shapeSize
        plt.figure()
        plt.imshow(image)
        return

    
    def makeTestBatch(self, selected_shape, n_shapes, batch_size, stim_idx=None,
                      centralize=False, reduce_df=False):
        '''
        Create one batch of the test dataset for the condition chosen with stim_idx.
        
        Parameters
        ----------
        selected_shape: int
                        Number corresponding to the shapeID, you want to use.
                        Special combinations are pre-defined here (e.g. 412 being
                        a switching pattern of shapeIDs 1 and 2)
        n_shapes: list of ints
                  This list should involve all possible shape repetitions (e.g.
                  [1, 3, 5])
        batch_size: int
                    chosen batch size
        stim_idx: int
                  Based on the stim_idx, a condition can be chosen. If stim_idx=None,
                  a random condition is used. If stim_idx=0 the vernier-alone
                  condition is chosen; if stim_idx=1 the crowding condition is
                  chosen (=single flanker condition); if stim_idx=2 the uncrowding
                  condition is chosen (multiple flankers condition either using
                  flankers of one or two different types); if stim_idx=3 the
                  control condition is chosen (no-crowding condition due to
                  sufficient spacing between flankers and vernier)
        centralize: bool
                    Place shapes right in the center of the image
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
        nshapeslabels: 1d vector
                       Vector that involves full batch of shape repetitions for 
                       the flanker stimulus
        nshapeslabels_idx: 1d vector
                           Corresponding to nshapeslabels, this vector involves
                           all corresponding indices (e.g. if there were only 1
                           and 5 shape repetitions, the corresponding indices
                           would be 0 and 1)
        x_vernier: 1d vector
                   This vector involves a full batch of x-coordinates corresponding
                   to the upper left corner of the vernier patch
        y_vernier: 1d vector
                   Same for y-coordinate
        x_shape: 1d vector
                 This vector involves a full batch of x-coordinates corresponding
                 to the upper left corner of the most left shape patch (= first
                 shape of the group)
        y_shape: 1d vector
                 This vector involves a full batch of y-coordinates corresponding
                 to the upper left corner of the most left shape patch (= first
                 shape of the group)
        '''

        # Initializations
        vernier_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shape_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        
        for idx_batch in range(batch_size):
            
            vernier_image = np.zeros(self.imSize, dtype=np.float32)
            shape_image = np.zeros(self.imSize, dtype=np.float32)
            if stim_idx is None:
                idx = np.random.randint(0, 3)
            else:
                idx = stim_idx
            
            if centralize:
                # Put each shape in the center of the image:
                row = int((self.imSize[0] - self.shapeSize) / 2)
            else:
                # Get a random y-coordinate and vernier offset direction for the vernier:
                row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(shapeID=0, offset_direction=offset_direction)

            # the 4XY-category is for creating X-Y configurations
            # (so-called no-uncrowding stimuli)
            if selected_shape==412:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==421:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==413:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==431:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==414:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=4)
            elif selected_shape==441:
                shape_patch = self.drawShape(shapeID=4)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==415:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=5)
            elif selected_shape==451:
                shape_patch = self.drawShape(shapeID=5)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==416:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=6)
            elif selected_shape==461:
                shape_patch = self.drawShape(shapeID=6)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==423:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==432:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==424:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=4)
            elif selected_shape==442:
                shape_patch = self.drawShape(shapeID=4)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==425:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=5)
            elif selected_shape==452:
                shape_patch = self.drawShape(shapeID=5)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==426:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=6)
            elif selected_shape==462:
                shape_patch = self.drawShape(shapeID=6)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==434:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=4)
            elif selected_shape==443:
                shape_patch = self.drawShape(shapeID=4)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==435:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=5)
            elif selected_shape==453:
                shape_patch = self.drawShape(shapeID=5)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==436:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=6)
            elif selected_shape==463:
                shape_patch = self.drawShape(shapeID=6)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==445:
                shape_patch = self.drawShape(shapeID=4)
                uncrowding_patch = self.drawShape(shapeID=5)
            elif selected_shape==454:
                shape_patch = self.drawShape(shapeID=5)
                uncrowding_patch = self.drawShape(shapeID=4)
            elif selected_shape==446:
                shape_patch = self.drawShape(shapeID=4)
                uncrowding_patch = self.drawShape(shapeID=6)
            elif selected_shape==464:
                shape_patch = self.drawShape(shapeID=6)
                uncrowding_patch = self.drawShape(shapeID=4)
            elif selected_shape==456:
                shape_patch = self.drawShape(shapeID=5)
                uncrowding_patch = self.drawShape(shapeID=6)
            elif selected_shape==465:
                shape_patch = self.drawShape(shapeID=6)
                uncrowding_patch = self.drawShape(shapeID=5)
            else:
                shape_patch = self.drawShape(shapeID=selected_shape)
            
            # The different test conditions:
            if idx==0:
                # Vernier-only test stimuli:
                selected_repetitions = 0
                nshapes_label = 0
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize) / 2)
                else:
                    if reduce_df:
                        # We want to make the degrees of freedom for position on
                        # the x axis fair:
                        imSize_adapted = self.imSize[1] - (max(n_shapes)-1)*self.shapeSize
                        imStart = int((self.imSize[1] - imSize_adapted) / 2)
                        col = np.random.randint(imStart, imStart+imSize_adapted - self.shapeSize*1)

                    else:
                        col = np.random.randint(0, self.imSize[1] - self.shapeSize)

                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==1:
                # Crowded test stimuli / Single flanker stimuli:
                selected_repetitions = 1
                nshapes_label = n_shapes.index(selected_repetitions)
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize) / 2)
                else:
                    if reduce_df:
                        # We want to make the degrees of freedom for position on
                        # the x axis fair:
                        imSize_adapted = self.imSize[1] - (max(n_shapes)-1)*self.shapeSize
                        imStart = int((self.imSize[1] - imSize_adapted) / 2)
                        col = np.random.randint(imStart, imStart+imSize_adapted - self.shapeSize*1)

                    else:
                        col = np.random.randint(0, self.imSize[1] - self.shapeSize)
        
                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==2:
                # Uncrowding / No-uncrowidng test stimuli:
                selected_repetitions = np.max(n_shapes)
                nshapes_label = n_shapes.index(selected_repetitions)
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize*selected_repetitions) / 2)
                else:
                    if reduce_df:
                        # We want to make the degrees of freedom for position on
                        # the x axis fair:
                        imSize_adapted = self.imSize[1] - (max(n_shapes)-selected_repetitions)*self.shapeSize
                        imStart = int((self.imSize[1] - imSize_adapted) / 2)
                        col = np.random.randint(imStart, imStart+imSize_adapted - self.shapeSize*selected_repetitions)

                    else:
                        col = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
    
                x_shape_ind, y_shape_ind = col, row
                
                if (selected_repetitions-1)/2 % 2 == 0:
                    trigger = 0
                else:
                    trigger = 1

                for n_repetitions in range(selected_repetitions):
                    if selected_shape>=400:
                        if n_repetitions == (selected_repetitions-1)/2:
                            vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                            x_vernier_ind, y_vernier_ind = col, row
                        if trigger == 0:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                            col += self.shapeSize
                            trigger = 1
                        else:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += uncrowding_patch
                            col += self.shapeSize
                            trigger = 0

                    else:
                        shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                        if n_repetitions == (selected_repetitions-1)/2:
                            vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                            x_vernier_ind, y_vernier_ind = col, row
                        col += self.shapeSize

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


    def makeTrainBatch(self, shape_types, n_shapes, batch_size, train_procedure='vernier_shape',
                       overlap=False, centralize=False, reduce_df=False):
        '''
        Create one batch of the training dataset with each two groups of shape_type
        repeated n_shapes[random] times
        
        Parameters
        ----------
        shape_types: list or int
                     Either list including all possible shapeIDs of which one ID
                     is randomly chosen to create the stimulus, or just a single 
                     shapeID that is used to create the stimulus.
        n_shapes: list of ints
                  This list should involve all possible shape repetitions (e.g.
                  [1, 3, 5])
        batch_size: int
                    chosen batch size
        train_procedure: string 
                         This input can be used to change the actual training input
                         of the network. To recapitulate the paper results, the
                         only training procedure that is needed is called 'random'.
                         If 'random', the network will receive an input
                         comprised of a single shape type which is in 50% of the cases
                         a vernier, and in the other 50% a random other group of
                         shapes (e.g. three squares). For this, shape_2_images will
                         be replaced by a noise image in the capser_input_fn.
                         If 'random_random', the network will receive an input
                         comprised of two shape types involving a vernier plus a
                         random group of shapes in 50% of the cases, or else two
                         random groups of shapes. If 'vernier_shape', the network
                         will receive an input which is always comprised of a vernier
                         and a random group of shapes. The default is 'random'.
        overlap: bool
                 If true, allow overlap between shape_1 and shape_2
        centralize: bool
                    Place shapes right in the center of the image
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
                        Array that involves full batch of gray-scale images only
                        including the flanker stimuli
        shapelabels_idx: 2d array
                         Array that involves full batch of shapeIDs for the vernier
                         and flanker stimulus
        vernierlabels_idx: 1d vector
                           Vector that involves full batch of vernier offset
                           directions
        nshapeslabels: 2d array
                       Array that involves full batch of shape repetitions for 
                       both images
        nshapeslabels_idx: 2d array
                           Corresponding to nshapeslabels, this array involves
                           all corresponding indices (e.g. if there were only 1
                           and 5 shape repetitions, the corresponding indices
                           would be 0 and 1)
        x_shape_1: 1d vector
                   This vector involves a full batch of x-coordinates corresponding
                   to the upper left corner of the vernier patch
        y_shape_1: 1d vector
                   Same for y-coordinate
        x_shape_2: 1d vector
                   This vector involves a full batch of x-coordinates corresponding
                   to the upper left corner of the most left shape patch (= first
                   shape of the group)
        y_shape_2: 1d vector
                   This vector involves a full batch of y-coordinates corresponding
                   to the upper left corner of the most left shape patch (= first
                   shape of the group)
        '''

        # Initializations
        shape_1_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shape_2_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        x_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            shape_1_image = np.zeros(self.imSize, dtype=np.float32)
            shape_2_image = np.zeros(self.imSize, dtype=np.float32)

            try:
                # Based on the train_procedure, the training stimuli will be
                # presented differently
                if train_procedure=='vernier_shape':
                    # Always have two stimuli in one image: vernier and another
                    # shape type
                    selected_shape_1 = 0
                    selected_shape_2 = np.random.randint(1, len(shape_types))
                elif train_procedure=='random_random' or 'random':
                    # Either have two random shape types in one image or only
                    # have one shape type per image (for this, we get rid of
                    # shape_2 in the capser_input_fn.py)
                    # Constraint: present vernier in 50% of the cases
                    if np.random.rand(1)<0.5:
                        selected_shape_1 = 0
                    else:
                        selected_shape_1 = np.random.randint(0, len(shape_types))
                    selected_shape_2 = np.random.randint(1, len(shape_types))
                else:
                    raise SystemExit('\nThe chosen train_procedure is unknown!\n')

            except:
                # If only one shape_type was given, use it
                selected_shape_1 = 0
                selected_shape_2 = shape_types

            
            # Create shape images:
            if selected_shape_1==0:
                # if the shape_1 is a vernier, only repeat once
                selected_repetitions_1 = 1
                idx_n_shapes_1 = n_shapes.index(selected_repetitions_1)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(shapeID=selected_shape_1, offset_direction=offset_direction)
            else:
                # if not, repeat shape random times but at least once
                idx_n_shapes_1 = np.random.randint(1, len(n_shapes))
                selected_repetitions_1 = n_shapes[idx_n_shapes_1]
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(shapeID=selected_shape_1)

            idx_n_shapes_2 = np.random.randint(0, len(n_shapes))
            selected_repetitions_2 = n_shapes[idx_n_shapes_2]
            shape_2_patch = self.drawShape(shapeID=selected_shape_2)

            if centralize:
                # Put each shape in the center of the image:
                row_shape_1 = int((self.imSize[0] - self.shapeSize) / 2)
                col_shape_1_init = int((self.imSize[1] - self.shapeSize*selected_repetitions_1) / 2)
                col_shape_1 = col_shape_1_init
                row_shape_2 = int((self.imSize[0] - self.shapeSize) / 2)
                col_shape_2_init = int((self.imSize[1] - self.shapeSize*selected_repetitions_2) / 2)
                col_shape_2 = col_shape_2_init
            else:
                row_shape_1 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                row_shape_2 = np.random.randint(0, self.imSize[0] - self.shapeSize)

                if reduce_df:
                    # We want to make the degrees of freedom for position on
                    # the x axis fair:
                    imSize_adapted_1 = self.imSize[1] - (max(n_shapes)-selected_repetitions_1)*self.shapeSize
                    imStart_1 = int((self.imSize[1] - imSize_adapted_1) / 2)
                    col_shape_1_init = np.random.randint(imStart_1, imStart_1+imSize_adapted_1 - self.shapeSize*selected_repetitions_1)
                    col_shape_1 = col_shape_1_init
                    imSize_adapted_2 = self.imSize[1] - (max(n_shapes)-selected_repetitions_2)*self.shapeSize
                    imStart_2 = int((self.imSize[1] - imSize_adapted_2) / 2)
                    col_shape_2_init = np.random.randint(imStart_2, imStart_2+imSize_adapted_2 - self.shapeSize*selected_repetitions_2)
                    col_shape_2 = col_shape_2_init
                else:
                    col_shape_1_init = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions_1)
                    col_shape_1 = col_shape_1_init
                    col_shape_2_init = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions_2)
                    col_shape_2 = col_shape_2_init
            
            # Repeat shape_2 selected_repetitions times:
            for i in range(selected_repetitions_2):
                shape_2_image[row_shape_2:row_shape_2+self.shapeSize,
                              col_shape_2:col_shape_2+self.shapeSize] += shape_2_patch
                col_shape_2 += self.shapeSize

            # Do we allow for overlap between vernier and shape image?
            if overlap or centralize:
                # shape_1 and shape_2 can be at the same positions
                    if selected_shape_1==0:
                        shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                      col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                    else:
                        for i in range(selected_repetitions_1):
                            shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                          col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                            col_shape_1 += self.shapeSize
            else:
                # shape_1 and shape_2 have to be at entirely different positions
                # Note: images have to be large enough
                counter = 0
                while np.sum(shape_2_image[row_shape_1:row_shape_1+self.shapeSize,
                                           col_shape_2_init:col_shape_2+self.shapeSize]) + np.sum(shape_1_patch) > np.sum(shape_1_patch):
                    counter += 1
                    if counter > 100000:
                        raise SystemExit('\nPROBLEM: cannot find a solution in '
                                         'which shape_1 and shape_2 do not overlap. '
                                         'Consider increasing the image_size')
                    
                    row_shape_1 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                    row_shape_2 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                
                    # It might be useful to move shape_2 too:
                    shape_2_image = np.zeros(shape=[self.imSize[0], self.imSize[1]], dtype=np.float32)
                    col_shape_2 = col_shape_2_init
                    for i in range(selected_repetitions_2):
                        shape_2_image[row_shape_2:row_shape_2+self.shapeSize,
                                      col_shape_2:col_shape_2+self.shapeSize] += shape_2_patch
                        col_shape_2 += self.shapeSize
                    
                    if reduce_df:
                        # We want to make the degrees of freedom for position on
                        # the x axis fair:
                        imSize_adapted_1 = self.imSize[1] - (max(n_shapes)-selected_repetitions_1)*self.shapeSize
                        imStart_1 = int((self.imSize[1] - imSize_adapted_1) / 2)
                        col_shape_1_init = np.random.randint(imStart_1, imStart_1+imSize_adapted_1 - self.shapeSize*selected_repetitions_1)
                        col_shape_1 = col_shape_1_init
                    else:
                        col_shape_1 = np.random.randint(0, self.imSize[1] - self.shapeSize)
                
                if selected_shape_1==0:
                    shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                  col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                    col_shape_1_init = col_shape_1
                else:
                    for i in range(selected_repetitions_1):
                        shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                      col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                        col_shape_1 += self.shapeSize
            

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
