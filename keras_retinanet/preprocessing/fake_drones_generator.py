"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import cv2


# ============================== Helper functions ==============================


def find_i_min_i_max(arr, axis, threshold):
    '''
    Purpose: Trimming
    '''

    sums = np.sum(arr, axis)

    i_min = 0
    while sums[i_min] < threshold:
        i_min += 1

    i_max = len(sums) - 1
    while sums[i_max] < threshold:
        i_max -= 1

    return i_min, i_max


def trim_image(img, threshold=0.1):

    alpha_ch = img[:, :, 3]
    i_min, i_max = find_i_min_i_max(alpha_ch, 0, threshold)
    j_min, j_max = find_i_min_i_max(alpha_ch, 1, threshold)

    return img[j_min:j_max, i_min:i_max]


def rotate_img(img, angle):

    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    cols_rot = int(rows * abs_sin + cols * abs_cos)
    rows_rot = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += (cols_rot - cols) / 2
    M[1, 2] += (rows_rot - rows) / 2

    result = cv2.warpAffine(img, M, (cols_rot, rows_rot))

    return trim_image(result)


def insert_subimg(img, subimg, row, col):

    assert img.shape[0] >= subimg.shape[0] + row
    assert img.shape[1] >= subimg.shape[1] + col

    result = np.copy(img)
    mask = np.stack((subimg[:, :, 3], subimg[:, :, 3], subimg[:, :, 3]), axis = 2)
    result[row:row + subimg.shape[0], col:col + subimg.shape[1]] *= (1 - mask)
    result[row:row + subimg.shape[0], col:col + subimg.shape[1]]  += mask * subimg[:, :, :3]

    blured = cv2.GaussianBlur(result[row:row + subimg.shape[0], col:col + subimg.shape[1]], (3,3), 0)
    result[row:row + subimg.shape[0], col:col + subimg.shape[1]] *= (1 - mask)
    result[row:row + subimg.shape[0], col:col + subimg.shape[1]] += mask * blured

    return result





def resize_img_and_bbox(img, bbox, shape):

    bbox[0] = int(shape[1] * bbox[0] / img.shape[1])
    bbox[1] = int(shape[0] * bbox[1] / img.shape[0])
    bbox[2] = int(shape[1] * bbox[2] / img.shape[1])
    bbox[3] = int(shape[0] * bbox[3] / img.shape[0])

    return cv2.resize(img, (shape[0], shape[1]) ), bbox


# =============================== The generator ==============================

class Drones_Cut_Paste_Generator(Generator):
    """ Generate drone web cut&paste dataset.


        assumption:
            "load_image" is called before "load_annotations"
            see generator.py at line: 300
    """

    def __init__(
        self,
        bgr_imgs, drone_imgs,
        bgr_indexes, drone_indexes,
        batch_size, batches_per_epoch,
        image_shape=(224, 224, 3),
        drone_size_range=(0.4, 0.6),
        drone_rotation_range=(-45, 45),
        # kwargs = None # may need something to pass to parent class
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.bgr_imgs = bgr_imgs
        self.drone_imgs = drone_imgs
        self.bgr_indexes = bgr_indexes
        self.drone_indexes = drone_indexes
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.example_cashe = {}  # save generated example configurations
        #                          such as used drone idx, background idx
        #                          and bounding boxes

        # augmentation parameters
        self.drone_size_range = drone_size_range
        self.drone_rotation_range = drone_rotation_range


        self.image_names = []
        self.image_data  = {}
        self.image_shape = image_shape


        # class for each annotation
        self.classes = {'drone' : 0} # a class index for each label
        self.labels  = {0 : 'drone'} # a name for each class

        # Temp variables
        self.bboxes = {}

        # super(Drones_Cut_Paste_Generator, self).__init__(**kwargs)
        super(Drones_Cut_Paste_Generator, self).__init__()


    # def decide_drone_positions(self, N_examples, size_range, angle_range):
    #     '''
    #     Decide drone bounding boxes in advance.
    #     TODO: implement if needed
    #     '''
    #     sizes  = np.random.uniform(*size_range, size = N_examples)
    #     angles = np.random.uniform(*angle_range, size = N_examples)


    # =============================== [img insert] ===========================
    def random_insert(self, img, subimg, size_range, angle_range, img_idx):


        min_size, max_size = size_range
        min_angle, max_angle = angle_range

        size = np.random.uniform(min_size, max_size)
        size = size * min(img.shape[0], img.shape[1])
        scale = size / max(subimg.shape[0], subimg.shape[1])


        self.example_cashe[img_idx]['scale'] = scale  # cashe for reproducability of idx


        subimg_resc = cv2.resize(subimg, (int(subimg.shape[1] * scale), int(subimg.shape[0] * scale)))

        angle = np.random.uniform(min_angle, max_angle)
        self.example_cashe[img_idx]['angle'] = angle

        subimg_resc = rotate_img(subimg_resc, angle)

        row = np.random.randint(img.shape[0] - subimg_resc.shape[0])
        col = np.random.randint(img.shape[1] - subimg_resc.shape[1])

        bbox = np.array([col, row, subimg_resc.shape[1], subimg_resc.shape[0]])  # x, y, w, h

        return insert_subimg(img, subimg_resc, row, col), bbox

    def deterministic_insert(self, img, subimg, img_idx):

        scale = self.example_cashe[img_idx]['scale']
        angle = self.example_cashe[img_idx]['angle']

        subimg_resc = cv2.resize(subimg,
                                 ( int(subimg.shape[1] * scale),
                                   int(subimg.shape[0] * scale) )  )

        subimg_resc = rotate_img(subimg_resc, angle)

        row = np.random.randint(img.shape[0] - subimg_resc.shape[0])
        col = np.random.randint(img.shape[1] - subimg_resc.shape[1])

        bbox = np.array([col, row, subimg_resc.shape[1], subimg_resc.shape[0]])  # x, y, w, h

        return insert_subimg(img, subimg_resc, row, col), bbox
    # =============================== [/img insert] ===========================

    def size(self):
        """ Size of the dataset.
        """
        # return len(self.image_names)
        return self.batches_per_epoch * self.batch_size

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    # def image_path(self, image_index):
    #     """ Returns the image path for image_index.
    #     """
    #     return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        # image = Image.open(self.image_path(image_index))
        # return float(image.width) / float(image.height)
        return float(self.image_shape[1]) / float(self.image_shape[0])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        # TODO: May need to change rgb to bgr

        # return read_image_bgr(self.image_path(image_index))
        #
        # X = np.empty((self.batch_size, *self.image_shape, 3), dtype='float32')
        # Y = np.empty((self.batch_size, 5), dtype='float32')
        #
        # example_cashe[image_index]

        if image_index in self.example_cashe.keys():
            # use images associated with image_index
            bgr_index   = self.example_cashe[image_index]['bgr_index']
            drone_index = self.example_cashe[image_index]['drone_index']

            bgr_img = np.divide(self.bgr_imgs[bgr_index], 255, dtype='float32')
            drone_img = np.divide( self.drone_imgs[drone_index], 255, dtype='float32')

            fake_img, bbox = self.deterministic_insert(
                bgr_img, drone_img, image_index)

        else:
            # chosen random drone and background imgages
            bgr_index = np.random.choice(self.bgr_indexes)
            drone_index = np.random.choice(self.drone_indexes)

            self.example_cashe[image_index] = {}
            self.example_cashe[image_index]['bgr_index'] = bgr_index
            self.example_cashe[image_index]['drone_index'] = drone_index

            bgr_img = np.divide(self.bgr_imgs[bgr_index], 255, dtype='float32')
            drone_img = np.divide(self.drone_imgs[drone_index], 255, dtype='float32')

            fake_img, bbox = self.random_insert(
                bgr_img, drone_img,
                self.drone_size_range,
                self.drone_rotation_range,
                image_index)



        fake_img, bbox = resize_img_and_bbox(fake_img, bbox, self.image_shape)

        self.bboxes[image_index] = bbox

        return fake_img


    def load_annotations(self, image_index):
        """ Load annotations for an image_index.

        """
        annotations = {}

        # annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        annotations['labels'] = np.array([0])  # allways a drone

        # get bounding box of the image
        # Eimantas format: x, y, width, height  --- it is at (top, left)
        x = float( self.bboxes[image_index][0] )
        y = float( self.bboxes[image_index][1] )
        w = float( self.bboxes[image_index][2] )
        h = float( self.bboxes[image_index][3] )

        annotations['bboxes'] = np.array([[
                                        x,
                                        y,
                                        x + w,
                                        y + h,
                                    ]])

        # -------------------------------- old ---------------------------------
        # path        = self.image_names[image_index]
        # annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        # for idx, annot in enumerate(self.image_data[path]):
        #     annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
        #     annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
        #         float(annot['x1']),
        #         float(annot['y1']),
        #         float(annot['x2']),
        #         float(annot['y2']),
        #     ]]))

        return annotations
