"""
calculate anchors with k means
"""

import numpy as np
from sklearn.cluster import KMeans
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import cv2

def get_bbox_normalized_width_height(label_txt_file, excluded_cls=None):
    """
    return bounding box numpy array in YOLO format:
    [x0, y0, w, h, cls]
    x0, y0, w, h: are normalized by image size and lie between (0,1)
    cls is the the cls index
    """
    wh = np.ones([1,2]) * -1
    with open(label_txt_file, 'r') as f:
        for line in f:
            vec = line.split(' ')
            cls = int(vec[0])
            if cls in excluded_cls:
                continue
            w = float(vec[3])
            h = float(vec[4])
            if wh.sum() < 0:
                wh = np.array([w, h])
            else:
                wh = np.vstack((wh, np.array([w, h])))

    if wh.sum() < 0:
        wh = np.array([])
    return wh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./YoloV3/data/merged_kitti_vehicle_train.txt',
                        help='text file containing all path files')
    parser.add_argument('--num-anchors', type=int, default=6,
                        help="number of classes")
    parser.add_argument('--exclude-cls', type=int, default=None,
                        nargs='+',
                        help='class index which will not be considered')
    parser.add_argument('--output-np-file', type=str, default='./YoloV3/data/all_kitti_bbox_wh.npy',
                        help='class index which will not be considered')
    parser.add_argument('--img-width', type=int, default=416,
                        help='width of input to cnn')
    parser.add_argument('--img-height', type=int, default=416,
                        help='width of input to cnn')
    parser.add_argument('--output-txt-file', type=str, default='./YoloV3/data/kitti_bbox_wh.txt',
                        help='class index which will not be considered')

    parser.add_argument('--load-bbox', type=str, default=None,
                        help='load previous bbox')
    args, unknow_args = parser.parse_known_args()

    # go over all files in list. create numpy array [width, height] for every bounding box

    if args.load_bbox:
        bbox_wh = np.load(args.load_bbox)
    else:
        bbox_wh = np.ones([1, 2]) * -1
        with open(args.data, 'r') as f:
            for img_path in tqdm(f, ncols=100, desc='retrieving bboxes'):
                img_path = img_path.split('\n')[0]
                _, file_ext = os.path.splitext(img_path)
                label_path = img_path.replace("images", "labels").replace(file_ext, ".txt")
                img = cv2.imread(img_path)
                H, W, _ = img.shape

                wh_normalized = get_bbox_normalized_width_height(label_path, args.exclude_cls)
                if len(wh_normalized) > 1:
                    if bbox_wh.sum() < 0:
                        bbox_wh = wh_normalized * np.array([W, H])
                    else:
                        bbox_wh = np.vstack((bbox_wh, wh_normalized * np.array([W, H])))

                np.save(args.output_np_file, bbox_wh)

        print('finshed reading bboxes from files')

    # finding mean and std of anchors
    mean_w,mean_h = np.mean((bbox_wh), axis=0)
    std_w, std_h = np.std((bbox_wh), axis=0)

    bbox_wh = np.minimum(bbox_wh, 416 * np.ones_like(bbox_wh))

    # bbox_wh = bbox_wh[(abs(bbox_wh[:, 0] - mean_w) < std_w) * (abs(bbox_wh[:, 1] - mean_h) < std_h), :]

    bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]

    bbox_ratio = bbox_wh[:, 0] / bbox_wh[:, 1]
    bbox_ratio = bbox_ratio.reshape(-1, 1)

    plt.figure(0)
    plt.scatter(bbox_wh[:, 0], bbox_wh[:, 1], alpha=0.5, s=0.1)
    plt.xlabel('width[ps]')
    plt.ylabel('height[ps]')
    plt.title('anchors height Vs width')

    plt.figure(2)
    plt.scatter(bbox_ratio, bbox_area, alpha=0.5, s=0.1)
    plt.xlabel('ratio [w/h][ps]')
    plt.ylabel('area')
    plt.title('bbox area vs ratio')


    # performing k means with with number of clusters which was given as input
    kmeans = KMeans(n_clusters=args.num_anchors, verbose=1).fit(bbox_wh)
    anchors = kmeans.cluster_centers_
    print('predicted anchor w,h:')
    print(np.array(anchors).astype(np.int))
    plt.figure(0)
    plt.scatter(anchors[:, 0], anchors[:, 1], alpha=0.5, s=10)

    # performing k means on ratio
    kmeans = KMeans(n_clusters=3, verbose=0).fit(bbox_ratio)
    anchors_ratio = kmeans.cluster_centers_
    print('predicted anchor ratio:')
    print(anchors_ratio)

    with open(args.output_txt_file, 'w') as f:
        for item in anchors:
            f.write("%s\n" % item)
    print('saved anchors to: {}'.format(args.output_txt_file))



    anchors = np.array(anchors).astype(np.int)
    img = np.zeros((416,416))

    for _, a in enumerate(anchors):
        img = cv2.rectangle(img, (200 - int(a[0]/2), 200 - int(a[1]/2)),
                                 (200 + int(a[0] / 2), 200 + int(a[1] / 2)),
                                  color=(255,255,255),
                            thickness=2)

    plt.figure(3)
    plt.imshow(img)
    plt.show()



