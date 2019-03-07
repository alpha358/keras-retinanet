# ============================================================================ #
#                                    IMPORTS                                   #
# ============================================================================ #

import numpy as np
import time
import tqdm
import cv2
import pandas as pd
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ============================================================================ #
#                                     TOOLS                                    #
# ============================================================================ #
def display_imgs(imgs):
    columns = 3
    rows = 3
    img_nums = np.random.choice(len(imgs), columns * rows)
    img_data = imgs[img_nums]

    fig = plt.figure(figsize=(columns * 5, rows * 4))
    for i in range(rows):
        for j in range(columns):
            idx = i + j * columns
            fig.add_subplot(rows, columns, idx + 1)
            plt.axis('off')
#             img = img_data[idx].astype(np.float32)
            img = img_data[idx]
            plt.imshow(img)

    plt.tight_layout()
    plt.show()


def plot_history(history):
    plt.figure(figsize=(8, 10))

    plt.subplot(311)
    plt.plot(history.history['loss'])
    plt.plot(history.history['regression_loss'])
    plt.plot(history.history['classification_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'regression_loss', 'classification_loss'])

    plt.subplot(312)
    plt.plot(history.history['mAP'])
    plt.title('mAP')
    plt.xlabel('epoch')

    plt.subplot(313)
    plt.plot(history.history['lr'])
    plt.title('learning rate')
    plt.xlabel('epoch')

    plt.subplots_adjust(hspace=0.5)

    plt.show()

# ============================================================================ #
#                                 LOAD DATASETS                                #
# ============================================================================ #

# ------------------------------ helper functions ------------------------------
def load_images(paths, N_max=1000, alpha_chnl=False):
    '''
        Purpose: load images from paths.
                    for drones use alpha channel
    '''

    if len(paths) > N_max:
        paths = paths[0: N_max]

    # imgs = np.zeros((len(paths), *img_shape, 3), dtype=np.uint8)
    imgs = []

    for num, img_path in enumerate(paths):
        if alpha_chnl:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return np.array(imgs)


def read_file_paths(file_name):
    '''
    Purpose: Read file paths from data frame
    '''

    data = pd.read_csv(file_name)
    paths = data['path']

    return np.array(paths)


def get_image_paths(folder):
    '''
    Purpose: get image paths from a folder
    Return: a list of paths
    '''
    # df = pd.DataFrame(columns=['path'])
    paths = []
    for i, fname in enumerate(os.listdir(folder)):
        if fname[-3:] in ['png', 'jpg', 'JPG', 'PNG']:
            # df.loc[i] = [os.path.join(folder, fname)]
            paths.append(os.path.join(folder, fname))

    return paths


# ------------------------------- main function --------------------------------

def load_drone_dataset(bgrs_path, drones_path, N_max=1000, new_split=True, val_ratio=0.2):
    '''
    Purpose: drone and background images for
            generation of examples

    Input:
        bgrs_path   --- dir of backgrounds
        drones_path --- dir of drones
        new_split   --- new selection of images from the dataset
                        to train, test sets

    Output:
        train_drone_imgs  --- (example_idx, wx, wy, chnl = 4)
        train_bgr_imgs    --- (example_idx, wx, wy, chnl = 3)
        test_drone_imgs   --- (example_idx, wx, wy, chnl = 4)
        test_bgr_imgs     --- (example_idx, wx, wy, chnl = 3)

    '''
    def limit_number_of_images(paths, n_max=N_max):
        if len(paths) > n_max:
            paths = paths[0: n_max]
        return paths

    # -------------------------- loading image paths ---------------------------
    if new_split:
        bgr_paths = limit_number_of_images(get_image_paths(bgrs_path))
        drone_paths = limit_number_of_images(get_image_paths(drones_path))
        print('Number of background images:', len(bgr_paths))
        print('Number of drone images:', len(drone_paths))

        def split_fn(paths):
            '''Split paths'''
            return train_test_split(np.arange(len(paths)),
                                    test_size=val_ratio,
                                    random_state=24,
                                    shuffle=True)

        train_bgr_indexes, val_bgr_indexes = split_fn(bgr_paths)
        train_drone_indexes, val_drone_indexes = split_fn(drone_paths)
        print('Number of background images in validation set:', len(val_bgr_indexes))
        print('Number of drone images in validation set:', len(val_drone_indexes))
    # --------------------------- train_test split ----------------------------

    drone_images = load_images(drone_paths, N_max=1000, alpha_chnl=True)
    bgr_images = load_images(bgr_paths, N_max=1000, alpha_chnl=False)

    return drone_images, bgr_images, train_bgr_indexes, val_bgr_indexes, train_drone_indexes, val_drone_indexes





# ============================================================================ #
#                             GENERATE CSV DATASET                             #
# ============================================================================ #

def generate_csv_dataset(
    train_generator,
    val_generator,
    N_val,
    N_train,
    dir_name = 'drones_csv',
    ):

    '''
    Purpose: Generate a dataset with csv files for csv generator
    '''

    try:
        os.mkdir(dir_name) # dataset dir
        os.mkdir(os.path.join(dir_name, 'img')) # images dir
    except:
        pass

    # generate training examples
    #   --- assuming just one example drone per image
    for n in range(N_train):
        bbox = train_generator.load_image(n)