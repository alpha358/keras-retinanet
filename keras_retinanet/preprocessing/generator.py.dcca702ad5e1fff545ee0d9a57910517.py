"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import warnings
import keras
import cv2
import matplotlib.pyplot as plt

import imgaug as ia

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes,
    compute_missing_bbox_stats
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb
from ..utils.image import  cvt_grayscale
from ..utils.visualization import draw_boxes




def imnorm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# ------------------------------- imgaug tools ------------------------------- #
def to_imgaug_bboxes(bboxes, img_shape):
    '''
        Convert plain bboxes to imgaug bboxes.
        Input: []
        Output: []
    '''
    bboxes_list = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bboxes_list.append(
            ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        )

    return ia.BoundingBoxesOnImage(bboxes_list, shape=img_shape)


def to_plain_bboxes(bboxes_imgaug):
    '''
    Convert plain (initial) bboxes to imgaug format.
        Input: list of ia.BoundingBox
        Output: list of bbox tuples
    '''
    bboxes_list = []
    # for bbox in bboxes_imgaug.bounding_boxes:
    for bbox in bboxes_imgaug.bounding_boxes:
        # TODO: test
        x1, y1, x2, y2 = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.int)
        bboxes_list.append( (x1, y1, x2, y2) )

    return bboxes_list

# ------------------------------------- - ------------------------------------ #

# ---------------------------------------------------------------------------- #
#                                Generator Class                               #
# ---------------------------------------------------------------------------- #
class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
        compute_shapes=guess_shapes,
        preprocess_image=preprocess_image,
        config = None, # TODO: may become obsolete
        anchor_params = None,
        augmenter_imgaug = None, # imgaug augmenter
        grayscale = False,
        positive_overlap = 0.5,
        negative_overlap = 0.4,
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config
        self.augmenter_imgaug       = augmenter_imgaug
        self.grayscale              = grayscale
        self.anchor_params          = anchor_params
        self.positive_overlap = 0.5,
        negative_overlap = 0.4,


        # Use default anchor parameters if none are defined,
        #    use the params form config
        if self.anchor_params == None:
            if self.config and 'anchor_parameters' in self.config:
                self.anchor_params = parse_anchor_parameters(self.config)


        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')


    def generate_example(self, image_index):
        ''' Generate a sigle example.

            Return (img, annotations)
            Used in later eval.py
        '''
        image = self.load_image(image_index)
        annot = self.load_annotations(image_index)

        if self.grayscale:
            image = cvt_grayscale(image)

        return image, annot



    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                # warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                #     group[index],
                #     image.shape,
                #     annotations['bboxes'][invalid_indices, :]
                # ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """

        if self.augmenter_imgaug:
            # --------------------- apply transformation to image ------------------------ #
            # imgaug augmentation
            # image = apply_transform(transform, image, self.transform_parameters)
            augmenter_det = self.augmenter_imgaug.to_deterministic()

            # augmentation
            image = augmenter_det.augment_image(image) # buvo augment_images
            # Recasting image as np array after augmentation
            # CV2 sometimes fails to draw on the image after augmentation
            # More details: https: // stackoverflow.com/questions/49571138/cant-draw-box-when-call-cv2-rectangle
            image = np.array(image) # H1: this flips the horizontal axis in the image ?
            # image = np.flip(image, 1) # correct for the flip


            # ---------------------------- bboxes augmentation --------------------------- #
            # convert bboxes to imgaug format
            bboxes_imgaug = to_imgaug_bboxes(annotations['bboxes'].copy(), image.shape)

            # augment the bboxes
            bboxes_imgaug = augmenter_det.augment_bounding_boxes(bboxes_imgaug)

            # remove bboxes that are outside of the image
            bboxes_imgaug = bboxes_imgaug.remove_out_of_image().cut_out_of_image()
            # bboxes_imgaug = [bbox.remove_out_of_image().cut_out_of_image()
            #                  for bbox in bboxes_imgaug]  # .bounding_boxes


            # -------------------------- update the annotations -------------------------- #
            # annotations['bboxes'] = np.array(to_plain_bboxes(bboxes_imgaug))
            annotations['bboxes'] = to_plain_bboxes(bboxes_imgaug)

            # TODO: test for bbox distortion
            annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float)

            if self.grayscale:
                image = cvt_grayscale(image)

        else:
            # randomly transform both image and annotations
            if transform is not None or self.transform_generator:

                if transform is None:
                    transform = adjust_transform_for_image(next(
                        self.transform_generator), image, self.transform_parameters.relative_translation)

                # -------------------------- old style augmentation -------------------------- #
                # image augmentation
                image = apply_transform(transform, image, self.transform_parameters)
                # Transform the bounding boxes in the annotations.
                annotations['bboxes'] = annotations['bboxes'].copy()
                for index in range(annotations['bboxes'].shape[0]):
                    annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])


        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(
                image_group[index], annotations_group[index])

        return image_group, annotations_group


    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image) # mode='tf') # tf instead of caffe

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape, anchor_params=None):

        return anchors_for_shape(image_shape, anchor_params=self.anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes(),
            positive_overlap = self.positive_overlap,
            negative_overlap = self.negative_overlap,
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        group --- list of indices of images in a group
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

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

    def get_all_bboxes(self):
        '''
        Purpose: Get bboxes of all images in the generator
        '''
        boxes = []
        for n in range(self.size()):
            boxes.append(self.load_annotations(n)['bboxes'][0])

        return boxes

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.

        group --- a list for img indices in one group
        """
        group = self.groups[index] # H1: maybe here was the problem
        inputs, targets = self.compute_input_output(group)

        return inputs, targets


    # -------------------------- anchor box diagnostics -------------------------- #
    def get_missing_bbox_stats(self, n_max = 100, assign_missed=True):

        # ------------------------------ get bbox stats ------------------------------ #
        missed_box_count,\
             missed_box_overlaps = compute_missing_bbox_stats(self.anchor_params, self, n_max = n_max, assign_missed=assign_missed)

        # flatten list of lists
        flatten = lambda l: [item for sublist in l for item in sublist]

        overlaps = np.array(flatten(missed_box_overlaps)).flatten()

        # -------------------------------- plot stats -------------------------------- #

        plt.figure(figsize = (13,6))
        plt.subplot(1,2,1)
        plt.plot(missed_box_count)
        plt.title('Missed bbox count')

        # plt.subplot(1,2,2)
        # plt.hist(overlaps)
        # plt.title('Missed bbox overlaps')
        # plt.show()


        return missed_box_count, missed_box_overlaps



# ---------------------------------------------------------------------------- #
#                                     TESTS                                    #
# ---------------------------------------------------------------------------- #

def _grayscale_test(generator, n_example = 1):

    # ------------------------------- For training ------------------------------- #
    # Compute with augmentator effects
    inputs, targets = generator.compute_input_output( [n_example] )
    img = inputs[0]
    # bbox = targets[0]
    # draw_boxes(img, bbox, (255, 255, 0), thickness=1)
    plt.imshow(img)
    plt.title('compute_input_output (for training)')
    plt.show()

    plt.imshow( imnorm(img) )
    plt.title('compute_input_output (for training) [normalized]')
    plt.show()

    # ------------------------------ For Evaluation ------------------------------ #
    raw_image, annotations = generator.generate_example(n_example)

    plt.imshow(raw_image)
    plt.title('Image for evaluation')
    plt.show()


    image = generator.preprocess_image(raw_image.copy())
    # bbox = annotations['bbox']
    plt.imshow(image)
    plt.title('Image for evaluation [Preprocessed]')
    plt.show()
