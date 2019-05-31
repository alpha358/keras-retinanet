import pandas as pd
import zipfile
from keras_retinanet.utils.visualization import draw_box, draw_circle
from sklearn.metrics import accuracy_score
'''
Purpose:
    Compute model mean IOU
    Show model predictions, export as a gif
'''

# ============================================================================ #
#                                    IMPORTS                                   #
# ============================================================================ #

import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import time
from tqdm import tqdm
import cv2
import os
import glob
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


from keras_retinanet.utils.predictions_analysis import *

# ============================================================================ #
#                                SHOW DETECTIONS                               #
# ============================================================================ #

from matplotlib import animation
# from JSAnimation import IPython_display

# def plot_movie_js(image_array):
#     dpi = 75.0
#     xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
#     fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)

# #     fig = plt.figure(figsize=(12, 12), dpi=45)
#     im = plt.figimage(image_array[0])

#     def animate(i):
#         im.set_array(image_array[i])
#         return (im,)

#     anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
#     display(IPython_display.display_animation(anim))


# ============================================================================ #
#                                TOOLS                                         #
# ============================================================================ #
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


# ---------------------------- non-max suppresion ---------------------------- #
# Malisiewicz et al.
def non_max_suppression_fast(boxes, probs, overlapThresh):
    '''
    boxes --- bboxes [box_idx, coord_idx]
    probs --- probabilities of bboxes
    '''

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
#     idxs = np.argsort(y2)
    # Correct, sort the probabilities based on argsort in descending order (higher probabilities at the front of the list).
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum( x1[i], x1[idxs[:last]] )
        yy1 = np.maximum( y1[i], y1[idxs[:last]] )
        xx2 = np.minimum( x2[i], x2[idxs[:last]] )
        yy2 = np.minimum( y2[i], y2[idxs[:last]] )


        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        union = (area[i] + area[idxs[:last]]) - intersection

        # compute the ratio of overlap
        iou = intersection / (union + 1e-16)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(iou > overlapThresh)[0] )))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


# ----------------------------------- misc ----------------------------------- #
def inspect_frame_gt(generator, n_frame):
    '''
    Purpose: show generator ground truth bbox for n_frame
    '''
    plt.figure(figsize=(13, 13))
    draw = generator.load_image(n_frame)
    try:  # try if there are bboxes
        box = generator.load_annotations(n_frame)['bboxes'][0]
        draw_box(draw, box.astype(int), color=label_color(0), thickness=1)
    except:
        pass
    plt.imshow(draw)
    plt.show()


class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()  # retain local pointer to value
        return value                     # faster to return than dict lookup

# -------------------------- intersection over union ------------------------- #
def iou(box1, box2):
    '''
    Box format is: x1,y1,x2,y2 :thinking:
    '''

    # Intersection
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def estimate_mean_iou(model, generator, n_batches):

    mean_iou = 0.0
    n = 0

    for i in range(n_batches):

        X_test, Y_test = generator[0]
        Y = model.predict(X_test)

        for j in range(Y_test.shape[0]):

            if Y_test[j][0] == 0:
                continue

            box1 = Y_test[j][1:]
            box2 = Y[j][1:]
            mean_iou += iou(box1, box2)
            n += 1

    return mean_iou / n

# ----------------------------- confusion matrix ----------------------------- #
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# ============================================================================ #
#                                GET DETECTIONS                                #
# ============================================================================ #
def plot_detections(
        model_test,
        generator,
        p_thresh,
        iou_thresh,
        N_img = None, # n examples to process
        plot_here = False,
        savedir = None,
        N_ZIP = 100, # how many files to zip
        labels_to_names = {0: 'drone'},
        aux_annot = None, # auxiliary annotations
        aux_threshold = 0.5
    ):
    '''
    Purpose: plot best bounding boxes and save as jpg.

    '''
    # detections dict for
    detections_dict = {
        'im_name' : [],
        'x1' : [],
        'y1' : [],
        'x2' : [],
        'y2' : [],
        'p'  : [],
        'label'  : []
        }

    try:
        os.mkdir(savedir) # try to create savedir
    except:
        pass

    if N_img == None:
        N_img = generator.size()

    # loop over examples
    for img_idx in tqdm(range(N_img)):

        # load image
        image = generator.load_image(img_idx)

        # get image name --- assuming CSV generator
        img_name = os.path.basename(generator.image_path(img_idx))

        annotation_true = generator.load_annotations(img_idx)
        drone_exist_in_img = annotation_true['bboxes'].size > 0

        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        boxes, scores, labels = model_test.predict_on_batch(
            np.expand_dims(image, axis=0)
        )

        # correct for image scale
        boxes /= scale


        # Choose colors
        color_pred = label_color(0)
        color_true = label_color(1)
        color_aux = label_color(2)

        # Auxiliary annotations
        if aux_annot:
            if img_name in aux_annot.keys():
                p_aux, x_aux, y_aux = aux_annot[img_name]
                radius = 10
                if p_aux >= aux_threshold: # todo: change to just >
                    draw_circle(draw, (x_aux, y_aux), radius, color_aux, thickness=1)

        # True BBox
        if drone_exist_in_img:
            # True bbox
            draw_box(draw, annotation_true['bboxes'][0], color=color_true, thickness=1)

        # ---------------------- Iterate over network detections --------------------- #
        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            # Save detections
            if score >= 0:
                x1, y1, x2, y2 = box.astype( int )
                detections_dict['x1'].append( x1 )
                detections_dict['y1'].append( y1 )
                detections_dict['x2'].append( x2 )
                detections_dict['y2'].append( y2 )
                detections_dict['p'].append( score )
                detections_dict['label'].append( labels_to_names[label] )

            # plot boxes
            if score > p_thresh:
                draw_box(draw, box.astype(int), color=color_pred,
                         thickness=1)  # predicted box
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, box.astype(int), caption)

        if savedir:
            cv2.imwrite(os.path.join(
                savedir, str(img_idx) + '.jpg'), draw)

            if N_ZIP: # use zip
                if (img_idx+1) % N_ZIP == 0: # +1 due to zero indexing
                    zip_n = img_idx // N_ZIP # zip file number

                    # zip all contents of detections directory
                    zip_file = zipfile.ZipFile(
                            savedir+'_'+str(zip_n)+'.zip', 'w',
                            zipfile.ZIP_DEFLATED
                        )
                    zipdir(savedir, zip_file)

                    # Clean detections directory
                    files = glob.glob(
                                os.path.join(savedir, '*') )
                    for f in files:
                        os.remove(f)


        if plot_here:
            plt.figure(figsize=(12,12))
            plt.imshow(draw)
            plt.show()

    return detections_dict




def get_detections(
            model_test,
            validation_generator,
            labels_to_names,  # {0: 'drone'}
            N_img = None,
            save_plots = False,
            savedir = 'detections',
            get_img_array = False
            ):
    '''
    Purpose: Compute predictions, save bbox results

    Input:
        model_test              --- testing model
        validation_generator    --- generator
        labels_to_names         ---
        N_img = None            --- number of examples to analyse
        save_plots=False        --- save plots of detection to hdd
        savedir = 'detections'  --- dir to save plots

    Returns:
        iou_of_boxes --- iou of bboxes
        true_boxes   --- true bboxes
        pred_boxes   --- predicted bboxes
        probs        --- detection probs
    '''

    # try to make directory for detections
    try:
        os.mkdir(savedir)
    except:
        pass


    if N_img == None:
        # test all images in generator if not stated othervise
        N_img = validation_generator.size()

    if get_img_array:
        image_array = np.empty(shape=(N_img, 480, 640, 3), dtype=np.uint8)


    # for each examlple (img index as key)
    iou_of_boxes = defaultdict(list) # list of iou's for boxes (if drone is there)
    probs_of_boxes = defaultdict(list) # list of probabilities for bboxes
    true_boxes = defaultdict(list) # list of true bboxes
    pred_boxes = defaultdict(list) # list of predicted bboxes

    for img_idx in tqdm( range(N_img) ):
        # load image
        image = validation_generator.load_image(img_idx)
        annotation_true = validation_generator.load_annotations(img_idx)

        drone_exist_in_img = annotation_true['bboxes'].size > 0
        if drone_exist_in_img:
            # If drone exist in the frame
            true_boxes[img_idx].append( annotation_true['bboxes'][0] )

        # copy to draw on
        # draw = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)  # convert colors back --- not needed now
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # rescale true annotations --- not needed here
#         annotation_true['bboxes'] /= scale

        # process image
        t0 = time.time()
        # TODO: may be faster to predict on batch not by one image ?
        boxes, scores, labels = model_test.predict_on_batch(
            np.expand_dims(image, axis=0)
        )
        t1 = time.time() - t0

        # correct for image scale
        boxes /= scale


        # ---------------------- Iterate over network detections --------------------- #
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            #     for box, score in zip(boxes[0], scores[0]):

            # if score < 0: # strange, but by design of original developers
            #     # scores are sorted so we can break
            #     break

            probs_of_boxes[img_idx].append(score)



            color_pred = label_color(label)
            color_true = label_color(label+1)

            if drone_exist_in_img:
                iou_ = iou(annotation_true['bboxes'][0], box)
            else:
                iou_ = 0 # zero IoU if there is no drone

            iou_of_boxes[img_idx].append(iou_)

            # TODO: accuracy
            # detection --- at least one bbox detects the drone with p_thresh, iou_threshold

            pred_boxes[img_idx].append(box)

            # ---------------------------- plots bboxes on img --------------------------- #
            if (save_plots or get_img_array) and label >= 0:
                draw_box(draw, box.astype(int), color=color_pred,
                         thickness=1)  # predicted box
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, box.astype(int), caption)

                if drone_exist_in_img:
                    draw_box(draw, annotation_true['bboxes'][0], color=color_true, thickness=1)  # predicted box

                if get_img_array:
                    image_array[img_idx, :, :, :] = draw

                if save_plots:
                    cv2.imwrite(os.path.join(
                        savedir, str(img_idx) + '.jpg'), draw)


    # iou_of_boxes = np.array(iou_of_boxes)

    if get_img_array:
        return iou_of_boxes, true_boxes, pred_boxes, probs_of_boxes, image_array
    else:
        return iou_of_boxes, true_boxes, pred_boxes, probs_of_boxes


# ============================================================================ #
#                            GET PREDICTION VECTORS                            #
# ============================================================================ #

def get_prediction_vectors(
                pred_boxes,
                true_boxes,
                probs_of_boxes,
                iou_of_boxes,
                N_examples,
                drone_anywhere = False,
                iou_thresh = 0.5,
                prob_thresh = 0.5
                ):
    '''
    Purpose: Get the prediction vectors, y_pred, y_true
            given the assumptions about what counts as a prediction.

            y_hat = 1 if
                1) drone exists and is detected at the right place
                2) drone does not exist but is detected at the wrong place

    Input:
        pred_boxes     --- predicted bboxes, dict of lists
        true_boxes     --- true bboxes
        iou_of_boxes   --- IoU of boxes
        N_examples     --- number of examples in dataset, used for img_idx generation
        drone_anywhere --- dont care if drone is detected at wrong place
        iou_thresh     --- Intersection over union threshold
        prob_thresh    --- probability of bbox threshold

    Output:
        y_gt  --- ground truth
        y_hat --- predicted drone detections
        iou_of_confident_detections --- iou for bboxes where  p > p_thresh

    Warning: not tested for the case of many drones in the image.
    '''

    y_hat = []
    y_gt = []

    y_full_hat = []
    y_full_gt = []

    iou_of_confident_detections = []  # todo: may need to change to default dict
    '''
    y_hat[idx] --- 1 - network point to the right place for a drone
               --- 0 - network did not found the drone, or points to a wrong place

    y_gt[idx]  --- 1 - there is a drone in the image
               --- 0 - no drone in the image

    y_full_hat   --- the same but account for every confident bbox
    y_full_gt   --- the same but account for every confident bbox
    '''
    def arr(x): return np.array(x)


    for idx in range(N_examples):
        # iterating over images

        # logical vars
        drone_exist = len(true_boxes[idx]) > 0
        detections_exist = len(pred_boxes[idx]) > 0

        # --------------- logic of detection ---------------- #
        if drone_exist:
            y_gt.append(1)

            if detections_exist:
                confident_predictions = arr(probs_of_boxes[idx]) > prob_thresh
                good_iou_detections = arr(iou_of_boxes[idx]) > iou_thresh

                # IoU of confident predictions
                #   --- all predictions satisfying prob_thresh
                box_idx = 0


                # Account for each confident bbox
                # -------------------------------------------------------------
                if np.any(confident_predictions):
                    for confident in confident_predictions:
                        if confident:
                            # if confident and if iou is good enough
                            y_full_gt.append(1)
                            y_full_hat.append(iou_of_boxes[idx][box_idx] > iou_thresh)


                            iou_of_confident_detections.append(
                                iou_of_boxes[idx][box_idx]
                            )
                        box_idx += 1 # next box
                else:
                    # no confident preditions
                    y_full_gt.append(1)
                    y_full_hat.append(0)
                # -------------------------------------------------------------

                # Compute prediction vector
                if drone_anywhere:
                    # there is any detection with high prob
                    y_hat.append(int(np.any(confident_predictions)))
                else:
                    # drone is detected at the right place
                    # TODO: think weather this is okay
                    y_hat.append(
                        int(np.any(
                            confident_predictions * good_iou_detections > 0
                        ))
                    )
            else:
                # no detections, drone exist
                y_full_gt.append(1)
                y_full_hat.append(0)

        else:
            # drone does not exist
            y_gt.append(0)

            if detections_exist:
                confident_predictions = arr(probs_of_boxes[idx]) > prob_thresh

                # Account for each confident bbox
                # -------------------------------------------------------------
                if np.any(confident_predictions):
                    for confident in confident_predictions:
                        if confident:
                            y_full_gt.append(0)
                            y_full_hat.append(1)
                else:
                    # zero confident predictions
                    y_full_gt.append(0)
                    y_full_hat.append(0)
                # -------------------------------------------------------------

                # no need to check IoU for failed detections
                y_hat.append(int(np.any(confident_predictions)))
            else:
                # no detections, no drone
                y_hat.append(0)
                y_full_gt.append(0)
                y_full_hat.append(0)


    y_full_gt = arr(y_full_gt)
    y_full_hat = arr(y_full_hat)

    return y_gt, y_hat, y_full_gt, y_full_hat, arr(iou_of_confident_detections)

# ============================================================================ #
#                                    GET IOU                                   #
# ============================================================================ #
def get_iou(
        pred_boxes,
        true_boxes,
        probs_of_boxes,
        iou_of_boxes,
        iou_thresh = 0.5,
        p_thresh = 0.5
    ):
    '''
    Purpose: get IoU vector of proper detections
    '''

    return None


# ============================================================================ #
#             DETECTOR ONE-SHEET: MAX ACCURACY AND CONFUSION MATRIX            #
# ============================================================================ #
def detector_one_sheet(
    model, # testing model
    generator,
    labels_to_names,
    save_detection_images = False,
    report_dir = './report',
    N_img = None, # None means all images from generator
    lang = 'en', # plots language
    iou_thresh = 0.5,
    p_thresh = None,
    plot_here = True,
    aux_annot = None, # auxiliary annotations for comparison
    save_detections_csv = True,
    csv_fname = 'detections.csv',
    nms_iou = 0.3
    ):
    '''
        if nms_iou is not None
    '''

    # Try to create report directories
    os.mkdir(report_dir)
    detections_dir = os.path.join(report_dir, 'detections')
    os.mkdir(detections_dir)




    # Get detections for images given by generator
    iou_of_boxes, \
    true_boxes, \
    pred_boxes, \
    probs_of_boxes = get_detections(
        model,
        generator,
        labels_to_names,
        save_plots = False, # we get all detections at this point, so plot later
        get_img_array = False,
        N_img = N_img,
        savedir = os.path.join(report_dir, 'detections')
    )


    if save_detections_csv:
        table_csv = {
            'img_name':[],
            'x1':[],
            'y1':[],
            'x2':[],
            'y2':[],
            'p':[],
        }

        # iterate over images
        for img_idx, probs in probs_of_boxes.items():

            # get image name
            img_name = os.path.basename(generator.image_path(img_idx))

            # list of predicted bboxes for an image
            bboxes = pred_boxes[img_idx]

            # compute nms indices for a single image
            nms_indices = non_max_suppression_fast(
                                    np.array(pred_boxes[img_idx]),
                                    np.array(probs),
                                    overlapThresh = nms_iou
                                )

            # iterate over detections in image
            # for n in range(len(probs)):
            for n in nms_indices:
                if probs[n] >= 0:
                    # read bbox
                    x1, y1, x2, y2 = bboxes[n]

                    table_csv['img_name'].append(img_name)
                    table_csv['p'].append(probs[n])
                    table_csv['x1'].append(x1)
                    table_csv['y1'].append(y1)
                    table_csv['x2'].append(x2)
                    table_csv['y2'].append(y2)

            # over each image in img_idx
            # # ---------------------------- non-max suppresion ---------------------------- #
            # table_csv['img_name'] = np.array(table_csv['img_name'])[nms_indices]
            # table_csv['p'] = np.array(table_csv['p'])[nms_indices]
            # table_csv['x1'] = np.array(table_csv['x1'])[nms_indices]
            # table_csv['y1'] = np.array(table_csv['y1'])[nms_indices]
            # table_csv['x2'] = np.array(table_csv['x2'])[nms_indices]
            # table_csv['y2'] = np.array(table_csv['y2'])[nms_indices]
            # # ------------------------------------- . ------------------------------------ #

        pred_annotations_df = pd.DataFrame(table_csv)
        pred_annotations_df.to_csv(os.path.join(report_dir, csv_fname), index = False)



    # ------------------------- Vary probability treshold ------------------------ #
    # probs_ = np.array(list(probs_of_boxes.values())).reshape(-1)
    # probs = np.sort(probs_[probs_ > 0])


    # a function for shorter code
    def prediction_vectors(prob_thresh):
        return get_prediction_vectors(
            pred_boxes,
            true_boxes,
            probs_of_boxes,
            iou_of_boxes,
            N_img,
            drone_anywhere=False,
            iou_thresh = iou_thresh,
            prob_thresh=prob_thresh
        )


    p_thresholds = np.linspace(0.0, 1, 500)
    acc = []
    for prob_thresh in p_thresholds:
        # Get detection vectors y_true, y_pred:
        #   y_true --- ground truth, object exist in the image ?
        #   y_pred --- is object detected with given iou_threshold and p_threshold  ?
        y_true, y_pred, y_full_true, y_full_pred,\
             iou_of_confident_detections = prediction_vectors( prob_thresh)

        acc.append(accuracy_score(y_full_true, y_full_pred))

    acc = np.array(acc)
    # ------------------------------------- - ------------------------------------ #

    # Optimal prob. threshold
    idx_max_acc = np.argmax(acc)
    p_optimal = p_thresholds[idx_max_acc]

    y_true, y_pred, y_full_true, y_full_pred, \
         iou_of_confident_detections = prediction_vectors(p_optimal)
    # import ipdb
    # ipdb.set_trace()  # debugging starts here


    # ============================================================================ #
    #                                     PLOTS                                    #
    # ============================================================================ #
    plots_words = Vividict() # just a nested dict
    plots_words['lt']['prob_threshold'] = 'tikimybės riba, %'
    plots_words['en']['prob_threshold'] = 'probability threshold, %'
    plots_words['en']['acc'] = 'accuracy, %'
    plots_words['lt']['acc'] = 'tikslumas, %'


    # ------------------------- Plot prob. treshold curve ------------------------ #
    # plt.subplot(2,1,1)
    max_acc = np.max(acc)
    plt.plot(p_thresholds*100, acc*100)
    plt.xlabel(plots_words[lang]['prob_threshold'])
    plt.ylabel(plots_words[lang]['acc'])
    plt.title('P_optimal = %2.3f' % p_optimal + ', ACC_MAX = %2.2f' % max_acc)
    plt.grid(True)
    # plt.savefig('acc_vs_prob_threshold.png')
    plt.savefig(os.path.join(report_dir, 'acc_vs_prob_threshold.png'))
    if plot_here:
        plt.show()

    plt.hist(iou_of_confident_detections)
    plt.title('Mean IoU = %2.2f' %
              np.mean(iou_of_confident_detections))
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, 'mean_iou.png'))
    if plot_here:
        plt.show()

    # --------------------------- Predictions at frames -------------------------- #
    img_idx = list(range(len(y_pred)))
    plt.figure(figsize=(10,4))
    plt.scatter(img_idx, y_true, marker=',',
                alpha=0.8, color='b', label='y_true', s=1)
    plt.scatter(img_idx, np.array(y_pred)*0.94 + 0.02 , marker=',', # not cool, hackish solution
                alpha=0.8, color='r', label='any_correct(y_pred)', s=1)
    plt.legend()
    # plt.title('P_optimal = %2.3f' % p_optimal + ', ACC_MAX = %2.2f' % max_acc)
    # plt.savefig('detections.png')
    plt.savefig(os.path.join(report_dir, 'predictions_at_frames.png'))
    if plot_here:
        plt.show()


    # ------------------------- ---- Confusion matrix --- ------------------------ #
    # plt.subplot(1,2,1)
    plot_confusion_matrix(y_full_true, y_full_pred, np.array(
        ['No Drone', 'Drone']), normalize=True)
    plt.title('P_optimal = %2.3f' % p_optimal + ', ACC_MAX = %2.2f' % max_acc)
    plt.savefig(os.path.join(report_dir , 'optimal_confusion_norm.png'))
    if plot_here:
        plt.show()

    # plt.subplot(1,2,2)
    plot_confusion_matrix(y_full_true, y_full_pred,
     np.array(['No Drone', 'Drone']), normalize=False)
    plt.title('P_optimal = %2.3f' % p_optimal + ', ACC_MAX = %2.2f' % max_acc)
    plt.savefig(os.path.join(report_dir , 'optimal_confusion.png'))
    if plot_here:
        plt.show()

    # ------------ plot detections at optimal p_thresh_detection_plots ----------- #
    # if probability is specified, than override
    if p_thresh:
        p_thresh_detection_plots = p_thresh
    else:
        p_thresh_detection_plots = p_optimal


    if save_detection_images:
        # print('--- Saving detection images ---')
        # TODO: reuse saved bboxes from the first network run
        plot_detections(
            model,
            generator,
            p_thresh_detection_plots,
            iou_thresh = iou_thresh,
            N_img=N_img,  # n examples to process
            plot_here = False,
            savedir = detections_dir,
            labels_to_names = labels_to_names,
            aux_annot = aux_annot
        )


    # ---------------------------------------------------------------------------- #
    #                              Precision vs Recall                             #
    # ---------------------------------------------------------------------------- #
    # read annotations: true and predicted
    # pred_annotations_df = pd.read_csv('detections.csv') # allready computed
    true_annotations_df = pd.read_csv(generator.csv_data_file,
                                        names=[ 'img_name', 'x1', 'y1', 'x2', 'y2', 'class'])

    # Convert dataframes  to dictionaries for speed
    true_annotations, pred_annotations = get_detection_dictionaries(true_annotations_df, pred_annotations_df)

    # Compute the relevant detection statistics
    TN, TP, FN, FP, p_thresholds = compute_detection_stats_vs_p_thresh(pred_annotations, true_annotations)

    # save detections statistics
    detection_stats = {
        'TN': TN,
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'p_thresholds': p_thresholds
    }

    # save detections statistics
    with open(os.path.join(report_dir, 'stats.pickle'), mode='wb') as h:
        pickle.dump(detection_stats, h)


    plot_detection_analysis(TN, TP, FN, FP, p_thresholds, report_dir)


    return p_optimal
