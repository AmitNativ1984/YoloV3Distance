import numpy as np
import os
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    source_path = "/home/amit/Data/OID/OID/noGroup/train/labels_org"
    dest_path = "/home/amit/Data/OID/OID/noGroup/train/labels"

    all_label_files = os.listdir(source_path)

    cls_dict = {
                "Window": 0,
                "Door": 1,
                "Person": 2,
                "House": 3,
                "Car": 4
                }
    pbar = tqdm(all_label_files, ncols=100)
    pbar.set_description('files converted')
    for label_file in pbar:
        if label_file.endswith('.txt'):
            with open(os.path.join(source_path, label_file), 'r') as label:
                img_file = os.path.join(dest_path.replace("labels", "images"),label_file.replace(".txt", ".jpg"))
                if not os.path.isfile(img_file):
                    continue

                img = cv2.imread(img_file)

                H, W, _ = img.shape

                dest_label_file = os.path.join(dest_path, label_file)
                bbox = []
                for line in label:
                    line = line.split('\n')[0]
                    vec = line.split(' ')
                    cls = cls_dict[vec[0]]
                    xmin = float(vec[1])
                    ymin = float(vec[2])
                    xmax = float(vec[3])
                    ymax = float(vec[4])

                    x0 = np.mean([xmin, xmax]) / W
                    y0 = np.mean([ymin, ymax]) / H
                    w = (xmax - xmin) / W
                    h = (ymax - ymin) / H

                    # # validating bbox size:
                    # w = min(w, 2 * (1 - x0 - 0.001), 2 * x0)
                    # h = min(h, 2 * (1 - y0 - 0.001), 2 * y0)

                    new_line = ' '.join([str(cls), str(x0), str(y0), str(w), str(h)])
                    bbox.append(new_line)

                # writing file to dest
                with open(dest_label_file, 'w') as dest_file:
                    for line in bbox:
                        dest_file.write("%s\n" % line)


