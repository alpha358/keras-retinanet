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


def mean_iou(model_test, validation_generator, labels_to_names):
    '''
    '''
    # load image
    # image = read_image_bgr('000000008021.jpg')
    N_test_img = 200

    # image_array = np.empty(shape=(N_test_img, 480, 640, 3), dtype=np.uint8)

    iou_of_boxes = []

    for n in range(N_test_img):
        img_idx = n+200
        # load image
        image = validation_generator.load_image(img_idx)
        annotation_true = validation_generator.load_annotations(img_idx)

        if annotation_true['bboxes'].size == 0:
            # if there is no drone, than bbox is zero for zero overlap
            annotation_true['bboxes'] = np.array([0, 0, 0, 0])


        # copy to draw on
        draw = image.copy()
    #     draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # rescale true annotations
        annotation_true['bboxes'] /= scale

        # process image
        start = time.time()
        boxes, scores, labels = model_test.predict_on_batch(
            np.expand_dims(image, axis=0))
    #     boxes, scores = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        draw_box(draw, box_int, color=color)

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            #     for box, score in zip(boxes[0], scores[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            box_int = box.astype(int)

            iou_of_boxes.append(
                iou(annotation_true['bboxes'], box_int)
            )


            # draw_box(draw, box_int, color=color)  # predicted box

    #         draw_box(draw, b, color=1)

            # caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, box_int, caption)

        # image_array[n, :, :, :] = draw
    #         plt.figure(figsize=(10, 10))
    #         plt.axis('off')
    #         plt.imshow(draw)
    #         plt.show()

    iou_of_boxes = np.array(iou_of_boxes)
    return iou_of_boxes
