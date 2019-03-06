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
import tqdm
import cv2
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt



# ============================================================================ #
#                                SHOW DETECTIONS                               #
# ============================================================================ #

from matplotlib import animation
from JSAnimation import IPython_display

def plot_movie_js(image_array):
    dpi = 75.0
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)

#     fig = plt.figure(figsize=(12, 12), dpi=45)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))


# ============================================================================ #
#                                TOOLS                                         #
# ============================================================================ #

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
    xi2 = np.minimum(box1[0] + box1[2], box2[0] + box2[2])
    yi2 = np.minimum(box1[1] + box1[3], box2[1] + box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
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

    for n in tqdm.tqdm( range(N_img) ):
        img_idx = n
        # load image
        image = validation_generator.load_image(img_idx)
        annotation_true = validation_generator.load_annotations(img_idx)

        drone_exist_in_img = annotation_true['bboxes'].size > 0
        if drone_exist_in_img:
            # If drone exist in the frame
            true_boxes[img_idx].append( annotation_true['bboxes'][0] )

        # copy to draw on
        draw = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)  # convert colors back

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

            # scores are sorted so we can break
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
            if save_plots or get_img_array:
                draw_box(draw, box.astype(int), color=color_pred)  # predicted box
                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, box.astype(int), caption)

                if drone_exist_in_img:
                    draw_box(draw, annotation_true['bboxes'][0], color=color_true)  # predicted box

                if get_img_array:
                    image_array[n, :, :, :] = draw

                if save_plots:
                    cv2.imwrite(os.path.join(savedir, str(n) + '.jpg'), draw)


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
    iou_of_confident_detections = []  # todo: may need to change to default dict
    '''
    y_hat[idx] --- 1 - network point to the right place for a drone
               --- 0 - network did not found the drone, or points to a wrong place

    y_gt[idx]  --- 1 - there is a drone in the image
               --- 0 - no drone in the image
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

                for confident in confident_predictions:
                    if confident:
                        iou_of_confident_detections.append(
                            iou_of_boxes[idx][box_idx]
                        )
                    box_idx += 1 # next box

                # Compute prediction vector
                if drone_anywhere:
                    # there is any detection with high prob
                    y_hat.append(int(np.any(confident_predictions)))
                else:
                    # drone is detected at the right place
                    y_hat.append(int(np.any(
                        confident_predictions * good_iou_detections
                        )))

        else:  # drone does not exist
            y_gt.append(0)

            if detections_exist:
                confident_predictions = arr(probs_of_boxes[idx]) > prob_thresh

                # no need to check IoU for failed detections
                y_hat.append(int(np.any(confident_predictions)))

    return y_gt, y_hat, arr(iou_of_confident_detections)


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
    detection_img_dir = './detections',
    N_img = None, # None means all images from generator
    lang = 'en' # plots language
    ):

    # Get detections for images given by generator
    iou_of_boxes, \
    true_boxes, \
    pred_boxes, \
    probs_of_boxes = get_detections(
        model,
        generator,
        labels_to_names,
        save_plots = save_detection_images,
        get_img_array = False,
        N_img = N_img,
        savedir = detection_img_dir
    )

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
            iou_thresh=0.5,
            prob_thresh=prob_thresh
        )


    p_thresholds = np.linspace(0.0, 1, 500)
    acc = []
    for prob_thresh in p_thresholds:
        # Get detection vectors y_true, y_pred:
        #   y_true --- ground truth, object exist in the image ?
        #   y_pred --- is object detected with given iou_threshold and p_threshold  ?
        y_true, y_pred, iou_of_confident_detections = prediction_vectors(prob_thresh)

        acc.append( accuracy_score(y_true, y_pred) )

    acc = np.array(acc)
    # ------------------------------------- - ------------------------------------ #

    # Optimal prob. threshold
    idx_max_acc = np.argmax(acc)
    p_optimal = p_thresholds[idx_max_acc]

    y_true, y_pred, iou_of_confident_detections = prediction_vectors(p_optimal)
    import ipdb
    ipdb.set_trace()  # debugging starts here


    # ----------------------------------- Plots ---------------------------------- #
    plots_words = Vividict() # just a nested dict
    plots_words['lt']['prob_threshold'] = 'tikimybės riba, %'
    plots_words['en']['prob_threshold'] = 'probability threshold, %'
    plots_words['en']['acc'] = 'accuracy, %'
    plots_words['lt']['acc'] = 'tikslumas, %'


    # Plot prob. treshold curve
    # plt.subplot(2,1,1)
    plt.plot(p_thresholds*100, acc*100)
    plt.xlabel(plots_words[lang]['prob_threshold'])
    plt.ylabel(plots_words[lang]['acc'])
    plt.title('P_optimal = %2.3f' % p_optimal)
    # plt.savefig('acc_vs_prob_threshold.png')
    plt.show()

    plt.hist(iou_of_confident_detections)
    plt.title('$\langle IoU \rangle$ = %2.2f' %
              np.mean(iou_of_confident_detections))
    plt.show()

    # Predictions at frames
    img_idx = list(range(len(y_pred)))
    plt.figure(figsize=(10,4))
    plt.scatter(img_idx, y_true, marker='$=$',
                alpha=0.1, color='b', label='y_true')
    plt.scatter(img_idx, y_pred, marker='$-$',
                alpha=0.1, color='r', label='y_pred')
    plt.legend()
    plt.title('P_optimal = %2.3f' % p_optimal)
    # plt.savefig('detections.png')
    plt.show()


    # ---- Confusion matrix ---
    plt.subplot(1,2,1)
    plot_confusion_matrix(y_true, y_pred, np.array(
        ['No Drone', 'Drone']), normalize=True)
    plt.title('P_optimal = %2.3f' % p_optimal)

    plt.subplot(1,2,2)
    plot_confusion_matrix(y_true, y_pred, np.array(
        ['No Drone', 'Drone']), normalize=False)

    plt.title('P_optimal = %2.3f' % p_optimal)
    # plt.savefig('optimal_confusion.png')
    plt.show()
