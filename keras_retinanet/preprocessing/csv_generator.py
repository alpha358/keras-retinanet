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
import os


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        # Alfonsas: just convert floats to int here
        x1 = _parse( int(float(x1)), int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse( int(float(y1)), int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse( int(float(x2)), int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse( int(float(y2)), int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            # raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            print('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            print('skipping this value')
            continue

        if y2 <= y1:
            # raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
            print('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
            print('skipping this value')
            continue

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        base_dir=None,
        grayscale=False,
        bgr_to_rgb = False,
        augmenter = None, # non-geometric augmenter
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).

            augmenter --- non-geometric augmentation fn.
            grayscale --- convert image to 3 channel grayscale ?
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir
        self.augmenter   = augmenter

        self.grayscale   = grayscale
        self.bgr_to_rgb  = bgr_to_rgb

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        self.drop_nonexisting_images() # TODO: why one execution is not enough ?

        super(CSVGenerator, self).__init__(**kwargs)

    def drop_nonexisting_images(self):
        '''
        Purpose: drop not found images from image_names, image_data
        '''
        idx = 0
        e = enumerate(np.copy(self.image_names))
        # using enumeration insted of for loop
        #   due to stop of the loop when deleting elements
        while True:
            try:
                idx, im_name = next(e)
                if not os.path.isfile(os.path.join(self.base_dir, im_name)):
                    # img not found - removing that line
                    print('Img not found, droping: ', im_name)
                    del self.image_data[im_name]
                    # self.image_data.pop(im_name)
                    del self.image_names[idx]
            except StopIteration:
                break


        # while image_names.next():
        #     im_name = image_names[idx]

        #     # for im_name in image_names:  # copy to continue iterations
        #     if not os.path.isfile(os.path.join(self.base_dir, im_name)):
        #         # img not found - removing that line
        #         print('Img not found, droping: ', im_name)
        #         del self.image_data[im_name]
        #         # self.image_data.pop(im_name)
        #         del self.image_names[idx]
        #     idx += 1 # img index

        import pdb; pdb.set_trace()

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

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

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        img = read_image_bgr(self.image_path(image_index))

        if self.bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.grayscale:

            # Assume BGR order, convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img_gray = np.mean(img, axis = 2) # simple solution
            img[:, :, 0] = np.asarray(img_gray, dtype=np.uint8)
            img[:, :, 1] = np.asarray(img_gray, dtype=np.uint8)
            img[:, :, 2] = np.asarray(img_gray, dtype=np.uint8)

        if self.augmenter:
            img = self.augmenter(img)

        return img

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path        = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations



# ============================================================================ #
#                            COMBINE CSV GENERATORS                            #
# ============================================================================ #
# Combine two csv generators into one generator

class Combined_CSVGenerator(Generator):
    def __init__(
        self,
        generator1,
        generator2,
        augmenter = None,
        # reshufle = True,
        **kwargs
    ):
        # init attributes
        # combined generator sizes
        N1 = generator1.size()
        N2 = generator2.size()

        self.augmenter = augmenter # non-geometric augmenter

        self.size_ = N1 + N2
        self.generators = [generator1, generator2]

        # ------------------------------ generator_example_indices ----------------------------- #
        # Combined index for child generators

        self.generator_example_indices = {}

        for example_idx in range(0, N1+N2):
            # Compute subgenerator index and its example index
            if example_idx >= N1:
                generator_idx = 1
                subgenerator_idx = example_idx - N1
            else:
                generator_idx = 0
                subgenerator_idx = example_idx

            self.generator_example_indices[example_idx] = (generator_idx, subgenerator_idx)


        # # Reshuffle
        # if reshufle:
        #     selection_idx = np.random.shuffle(
        #         np.arange(0, N1 + N2)
        #     )
        # else:
        #     selection_idx = np.arange(0, N1 + N2)

        # pass other args to father class
        super(Combined_CSVGenerator, self).__init__(**kwargs)


    def size(self):
        """ Size of the dataset.
        """
        return self.size_

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return self.generators[0].num_classes()

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        return self.generators[0].has_label(label) or \
                    self.generators[1].has_label(label)

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return self.generators[0].has_name(name) or \
                    self.generators[1].has_name(name)


    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.generators[0].name_to_label(name)


    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.generators[0].label_to_name(label)


    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        generator_idx, example_idx = self.generator_example_indices[image_index]
        return self.generators[generator_idx].image_aspect_ratio(example_idx)


    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        generator_idx, example_idx = self.generator_example_indices[image_index]
        img = self.generators[generator_idx].load_image(example_idx)

        # augment image if function is passed, nongeometric !
        if self.augmenter:
            img = self.augmenter(img)

        return img


    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        generator_idx, example_idx = self.generator_example_indices[image_index]
        return self.generators[generator_idx].load_annotations(example_idx)
