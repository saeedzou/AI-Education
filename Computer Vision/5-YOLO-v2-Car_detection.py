# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image


# %%

def yolo_filter_boxes(boxes, box_confidence, box_class_probabilities, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

        Arguments:
            boxes -- tensor of shape (19, 19, 5, 4)
            box_confidence -- tensor of shape (19, 19, 5, 1)
            box_class_probs -- tensor of shape (19, 19, 5, 80)
            threshold -- real value, if [ highest class probability score < threshold],
                         then get rid of the corresponding box

        Returns:
            scores -- tensor of shape (None,), containing the class probability score for selected boxes
            boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
            classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

        Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
        For example, the actual output size of scores would be (10,) if there are 10 boxes.
        """
    box_scores = box_confidence * box_class_probabilities

    classes = tf.argmax(box_scores, axis=-1)
    scores = tf.reduce_max(box_scores, axis=-1)
    indices = (scores >= threshold)

    scores = tf.boolean_mask(scores, indices)
    classes = tf.boolean_mask(classes, indices)
    boxes = tf.boolean_mask(boxes, indices)

    return scores, boxes, classes


# %%


def compute_iou(box1, box2):
    (x11, y11, x21, y21) = box1
    (x12, y12, x22, y22) = box2

    x1, y1 = max(x11, x12), max(y11, y12)
    x2, y2 = min(x21, x22), min(y21, y22)
    print(x1, y1)
    print(x2, y2)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 - y12) - intersection_area
    iou = intersection_area / union_area
    print(intersection_area, union_area, iou)
    return iou


# %%

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.6):
    nms = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=tf.Variable(max_boxes, dtype='int32'),
        iou_threshold=iou_threshold
    )
    scores = tf.gather(scores, nms)
    boxes = tf.gather(boxes, nms)
    classes = tf.gather(classes, nms)
    return scores, boxes, classes


# %%

def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_xxy, box_wh, box_confidence, box_class_probs = yolo_outputs

