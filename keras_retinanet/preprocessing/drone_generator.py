# Imports

import numpy as np
import cv2

from keras.utils import Sequence
import imgaug as ia

# Configuration

BACKGROUND_SHAPE = (480, 640)

# **TODO:** write this method
def convert_bboxes_to_gt(bboxes):
    pass

# ## Data generator

# Base class for generators. Bounding boxes should be a python list of `ia.BoundingBoxesOnImage`

class DroneGenerator(Sequence):
    def __init__(self, images, bboxes, batch_size, augmenter=None, shuffle=False):
        self.images = images
        self.bboxes = bboxes
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _get_indexes(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        return indexes
    
    
    def _finalize_batch(self, X):
        #convert to grayscale
        for i in range(len(X)):
            img = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)  
            img = np.expand_dims(img, axis=-1)
            img = np.tile(img, 3)
            X[i] = img
            
        X = np.divide(X, 255, dtype=np.float32)
        
        return X
    
    
    def _augment_batch(self, X, bboxes=None):
        if bboxes is None:
            X = self.augmenter.augment_images(X)
            return X
        else:
            augmenter_det = self.augmenter.to_deterministic()
            X = augmenter_det.augment_images(X)
        
            bboxes = augmenter_det.augment_bounding_boxes(bboxes)
            bboxes = [bbox.remove_out_of_image().cut_out_of_image() for bbox in bboxes]
        
            return X, bboxes


class DroneTrainValGenerator(DroneGenerator):
    def __init__(self, images, bboxes, batch_size, augmenter=None, shuffle=False):
        super().__init__(images, bboxes, batch_size, augmenter, shuffle)
    
    def __getitem__(self, idx):
        indexes = self._get_indexes(idx)
        
        X = self.images[indexes]
        # bboxes is python list, not numpy array
        bboxes = [self.bboxes[i] for i in indexes]
                
        if self.augmenter is not None:
            X, bboxes = self._augment_batch(X, bboxes)
        
        X = self._finalize_batch(X)
        Y = convert_bboxes_to_gt(bboxes) # TODO: this method needs to be written
        
        return X, Y


class DroneTestGenerator(DroneGenerator):
    def __init__(self, images, batch_size):
        super().__init__(images, None, batch_size)
    
    def __getitem__(self, idx):
        indexes = self._get_indexes(idx)
        X = self.images[indexes]
        X = self._finalize_batch(X)
        
        return X
