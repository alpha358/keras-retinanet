'''
Purpose:
    Compute model mean IOU
    Show model predictions, export as a gif
'''

import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import time

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


def mean_iou(model_test,
                validation_generator,
                labels_to_names,
                N_img = None,
                boxes_plots=False, ):
    '''
    '''

    if N_img==None:
        # test all images in generator if not stated othervise
        N_test_img = validation_generator.size()

    if boxes_plots:
        image_array = np.empty(shape=(N_test_img, 480, 640, 3), dtype=np.uint8)

    iou_of_boxes = []
    true_boxes = []
    pred_boxes = []

    for n in range(N_test_img):
        img_idx = n
        # load image
        image = validation_generator.load_image(img_idx)
        annotation_true = validation_generator.load_annotations(img_idx)

        if annotation_true['bboxes'].size == 0:
            continue  # skip images where there is no drone

        # copy to draw on
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # rescale true annotations --- not needed here
#         annotation_true['bboxes'] /= scale

        # process image
        start = time.time()
        boxes, scores, labels = model_test.predict_on_batch(
            np.expand_dims(image, axis=0))
    #     boxes, scores = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

#         draw_box(draw, box_int, color=color)

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            #     for box, score in zip(boxes[0], scores[0]):

            # scores are sorted so we can break
            if score < 0.5:
                break

            color_pred = label_color(label)
            color_true = label_color(label+1)

            iou_of_boxes.append(
                iou(annotation_true['bboxes'][0], box)
            )
            true_boxes.append(annotation_true['bboxes'][0])
            pred_boxes.append(box)


        if boxes_plots:
            draw_box(draw, box.astype(int), color=color_pred)  # predicted box
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, box.astype(int), caption)
            draw_box(draw, annotation_true['bboxes'][0], color=color_true)  # predicted box

            image_array[n, :, :, :] = draw

    iou_of_boxes = np.array(iou_of_boxes)

    if boxes_plots:
        return iou_of_boxes, true_boxes, pred_boxes, image_array
    else:
        return iou_of_boxes, true_boxes, pred_boxes
