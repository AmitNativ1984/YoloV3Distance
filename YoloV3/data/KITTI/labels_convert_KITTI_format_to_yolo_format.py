import numpy as np
import os
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    source_path = "/home/amit/Data/KITTI_object_detection/kitti_single/training/label_2_kitti_format"
    dest_path = "/home/amit/Data/KITTI_object_detection/kitti_single/training/label_2_yolo_format"


    os.makedirs("/home/amit/Data/KITTI_object_detection/kitti_single/training/label_2_yolo_format", exist_ok=True)
    all_label_files = os.listdir(source_path)

    cls_type_dict = {
        "Car": 1,
        "Van": 1,
        "Truck": 1,
        "Tram": 1,
        "Pedestrian": 0,
        "Person_sitting": 0,
        "Cyclist": 0,
    }
    cls_count = np.zeros(np.size(np.unique(list(cls_type_dict.values()))))

    pbar = tqdm(all_label_files, ncols=100)
    pbar.set_description('files converted')
    for label_file in pbar:
        if label_file.endswith('.txt'):
            with open(os.path.join(source_path, label_file), 'r') as label:
                img_file = os.path.join(source_path.replace("label_2_kitti_format", "image_2"),label_file.replace(".txt", ".png"))
                if not os.path.isfile(img_file):
                    continue

                img = cv2.imread(img_file)

                H, W, _ = img.shape

                dest_label_file = os.path.join(dest_path, label_file)
                bbox = []
                for line in label:
                    line = line.split('\n')[0]
                    vec = line.split(' ')
                    if vec[0] not in list(cls_type_dict.keys()):
                        continue
                    cls = cls_type_dict[vec[0]]
                    xmin = float(vec[4])
                    ymin = float(vec[5])
                    xmax = float(vec[6])
                    ymax = float(vec[7])

                    # img_temp = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=2)
                    # cv2.imshow('img', img_temp)
                    # cv2.waitKey(1)

                    x0 = np.mean([xmin, xmax]) / W
                    y0 = np.mean([ymin, ymax]) / H
                    w = (xmax - xmin) / W
                    h = (ymax - ymin) / H

                    new_line = ' '.join([str(cls), str(x0), str(y0), str(w), str(h)])
                    bbox.append(new_line)

                if bbox == []:
                    bbox = [-1 -1 -1 -1 -1]
                # writing file to dest
                with open(dest_label_file, 'w') as dest_file:
                    for line in bbox:
                        dest_file.write("%s\n" % line)


