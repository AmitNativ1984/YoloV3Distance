import numpy as np

def distance_over_IoU_thresh(gt_bboxes, pred_bboxes, IoUth=0.95):
    """
    calculate distance difference for all bounding boxes if their IoU with ground truth is over 0.95
    :param gt_distance: numpy array of ground truth bboxes
    :param pred_distance: numpy array of pred bboxes
    :return:
    """

    gtDist = np.array([])
    predDist = np.array([])
    for pred_bbox in pred_bboxes:
        pred_dist = pred_bbox[4]
        for gt_bbox in gt_bboxes:
            gt_dist = gt_bbox[4]

            # check cls is the same:
            if pred_bbox[5] == gt_bbox[5]:
                iou = bbox_iou(pred_bbox[:4], gt_bbox[:4])
                if iou >= IoUth:
                    if gtDist.size == 0:
                        gtDist = np.array([gt_dist])
                    else:
                        gtDist = np.vstack((gtDist, np.array([gt_dist])))

                    if predDist.size == 0:
                        predDist = np.array([pred_dist])
                    else:
                        predDist = np.vstack((predDist, np.array(pred_dist)))


    return predDist, gtDist

def bbox_iou(boxA, boxB):
    """ calculate IoU between two boxes """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou