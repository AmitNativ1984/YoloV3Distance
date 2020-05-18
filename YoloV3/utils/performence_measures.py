import numpy as np

def distance_diff(gt_bbox, pred_bbox, IoU=0.95):
    """
    calculate distance difference for all bounding boxes if their IoU with ground truth is over 0.95
    :param gt_distance: numpy array of ground truth bboxes
    :param pred_distance: numpy array of pred bboxes
    :return:
    """