# ---------------------------------- Imports --------------------------------- #

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import cv2
from os.path import join
from glob import glob

from .image_insert_class import  ImageInserter



# ---------------------------------------------------------------------------- #
#                                  Main class                                  #
# ---------------------------------------------------------------------------- #

def _get_paths(images_path, extensions = ['*.jpg', '*.jpeg', '*.png'] ):
    '''
    Read all img paths
    images_path --- path to images
    extensions  --- list of extensions
    '''

    files = []
    for ext in extensions:
        files.extend(glob(join(images_path, ext)))

    return files

class DroneReportGenerator(Generator):
    '''
    A generator for drone report.
    Random drone & bird insertion into background.

    epoch_size --- number of elements in one epoch
    '''
    def __init__(
        self,
        background_paths,
        drone_paths,
        bird_paths,
        epoch_size,
        batch_size = 8,
        augmenter = None,
        n_batches = 20,
        augmenter_imgaug = None,
        transform_generator = None
        ):

        # Set image paths
        self.background_paths = background_paths
        self.drone_paths = drone_paths
        self.bird_paths = bird_paths
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.max_n_birds = 2
        self.augmenter_imgaug = augmenter_imgaug
        self.transform_generator = transform_generator
        transform_generator
        

        # Create inserter objects
        self.drone_inserter = ImageInserter(
                inserted_image_paths = self.drone_paths,
                size_range  = ( 7./100, 70./100),
                angle_range = ( -45.0, 45.0 ),
                p = 1.0,
                feathering = False,
                thermal = False,
                shuffle_colors = True)

        self.bird_inserter = ImageInserter(
                inserted_image_paths = self.bird_paths,
                size_range  = ( 7./100, 70./100),
                angle_range = ( -45.0, 45.0 ),
                p = 1.0,
                feathering = False,
                thermal = False,
                shuffle_colors = True)

        # images_bgr ---  numpy tensor [batch_idx, height, width, channel
        # images, bboxes = img_inserter.insert_images(images_bgr)
        # images, bboxes = img_inserter.insert_images(images_bgr)

    # ----------------------------- Mandatory methods ---------------------------- #
    def size(self):
        """ Size of the dataset.
        """
        return self.epoch_size

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return 1

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        return True

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return True

    def name_to_label(self, name):
        """ Map name to label.
        """
        return 0


    def label_to_name(self, label):
        """ Map label to name.
        """
        return 'drone'


    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.

        A FIXED PARAMETER IN OUR CASE
        """
        return 640.0 / 480.0



    def load_annotations(self, image_index):
        """ Load annotations for an image_index. [Unused]
        """
        return None

    def load_image(self, image_index):
        ''' Load image for an image_index.  [Unused]
        '''
        return None


    def generate_example(self):
        """ Generate a single example
            Taking random image and background.
        """

        # Select random background image
        bg_idx = np.random.randint(0, len(self.background_paths) - 1 )
        bg_img = cv2.imread(self.background_paths[bg_idx])
        images = [bg_img]


        # Insert birds
        n_birds = np.random.randint(0, self.max_n_birds)
        for n in range(n_birds):
            images, _ = self.bird_inserter.insert_images(images)

        # Insert drone
        images, bboxes = self.drone_inserter.insert_images(images)


        # ----------------------------- BBox annotations ----------------------------- #
        x1 = bboxes[0].bounding_boxes[0].x1_int
        y1 = bboxes[0].bounding_boxes[0].y1_int
        x2 = bboxes[0].bounding_boxes[0].x2_int
        y2 = bboxes[0].bounding_boxes[0].y2_int

        # return 0-th image
        return images[0], (x1, y1, x2, y2)





    ''' Annotations format

    annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

            for idx, annot in enumerate(self.image_data[path]):
                annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
                annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                    float(annot['x1']),
                    float(annot['y1']),
                    float(annot['x2']),
                    float(annot['y2']),
                ]]))
    '''


    def load_group(self, n_elements):
        '''
        Load a group of images and bboxes

        Input:
            n_elements --- number of elements in this group

        Returns:
            image_group --- a list of images
            annotations_group --- a list of annotations
        '''

        image_group = []
        annotations_group = []

        # Construct a batch
        for _ in range(n_elements):

            # load example
            img, (x1, y1, x2, y2) = self.generate_example()

            # append image
            image_group.append(img)

            # ---------------------------- append annotations ---------------------------- #
            # left for future possibility of many drone per image
            annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label('drone')]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ]]))

            annotations_group.append(annotations)



        return image_group, annotations_group

    # -------------------------- Overriding parent class ------------------------- #
    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        group --- list of indices of images in a group
        """
        # load images and annotations
        n_elements = len(group)
        image_group, annotations_group = self.load_group( n_elements )

        # image_group       = self.load_image_group(group)
        # annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    # ---------------------------------------------------------------------------- #
    #                                     hacks                                    #
    # ---------------------------------------------------------------------------- #
    def __len__(self):
        """
        Number of batches for generator.
        """
        return self.n_batches
        # return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.

        group --- a list for img indices in one group
        """
        group = np.random.randint(0, high=self.epoch_size, size=self.batch_size)

        # group = self.groups[index] # H1: maybe here was the problem
        inputs, targets = self.compute_input_output(group)

        return inputs, targets