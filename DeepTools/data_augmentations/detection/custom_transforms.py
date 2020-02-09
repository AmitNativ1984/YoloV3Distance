import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from urllib.request import urlopen
from albumentations import (BboxParams,
                            HorizontalFlip,
                            VerticalFlip,
                            Resize,
                            CenterCrop,
                            RandomCrop,
                            Crop,
                            MotionBlur,
                            Compose)

BOX_COLOR = (0, 0, 1)     # BGR format
TEXT_COLOR = (1, 1, 1)


"""
The coco format of a bounding box:          [x_min, y_min, width, height], e.g. [97, 12, 150, 200].

The pascal_voc format of a bounding box:    [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212]
"""

# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='albumentations', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['category_id']))


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=None, thickness=2, bbox_format='yolo', Normalized=True):
    if not color:
        color = BOX_COLOR

    if bbox_format == 'yolo':
        x0, y0, w, h = bbox
        x_min = x0-w/2.
        x_max = x0+w/2.
        y_min = y0 - h / 2.
        y_max = y0 + h / 2.

        if Normalized:
            x_min, x_max, y_min, y_max = int(x_min *img.shape[1]), int((x_min + w)*img.shape[1]), int(y_min * img.shape[0]), int((y_min + h) * img.shape[0])
        else:
            x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    else:
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name, Normalized=True, color=None, plot=False):
    if not color:
        color = BOX_COLOR

    img = annotations['image'].copy()
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for idx, bbox in enumerate(annotations['bboxes']):
        if bbox[0] <= 0:
            continue
        # if np.size(bbox) == 5:
        img = visualize_bbox(img, bbox[:4], int(annotations['category_id'][idx]), category_id_to_name, Normalized=Normalized, color=color)
        # else:
        #     img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name, Normalized=Normalized, color=color)


    if plot:
        cv2.imshow('image', img)
        cv2.waitKey(1)
    return img



if __name__ == "__main__":
    def download_image(url):
        data = urlopen(url).read()
        data = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    image = cv2.imread('/home/amit/Data/Kzir/Windows/images/Day_984.bmp')
    # Annotations for image 386298 from COCO http://cocodataset.org/#explore?id=386298
    annotations = {'image': image, 'bboxes': [(0.730859, 0.5427084999999999, 0.035156, 0.039583)],
                   'category_id': [0]}
    category_id_to_name = {0: '0', 18: 'dog'}
    visualize(annotations, category_id_to_name)

    # testing data augmentations
    aug = get_aug([MotionBlur(p=1, blur_limit=7)])
    augmented = aug(**annotations)
    visualize(augmented, category_id_to_name)
    plt.show()