import numpy as np
import os
import time
from tqdm import tqdm


if __name__ == "__main__":
    source_path = "./YoloV3/data/merged_kitti_vehicle_train.txt"

    cls_type_dict = {
                '0': "Person",
                '1': "Car",
                }
    cls_count = np.zeros(np.size(np.unique(list(cls_type_dict.values()))))

    with open(source_path, 'r') as dataset:
        for img_file in dataset:
            img_file = img_file.split('\n')[0]
            label_file = img_file.replace("images", "labels").replace(os.path.splitext(img_file)[-1], ".txt")
            with open(label_file, 'r') as label:
                bbox = []
                for line in label:
                    line = line.split('\n')[0]
                    vec = line.split(' ')
                    cls = vec[0]
                    cls_name = cls_type_dict[cls]
                    cls_count[int(cls)] += 1

    total_count = np.sum(cls_count)
    time.sleep(0.1)
    cls_dict = {
        "Person": cls_count[0],
        "Car": cls_count[1]
    }

    print('cls [#] \n', cls_dict)