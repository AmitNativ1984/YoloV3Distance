import numpy as np
import os
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    source_path = "/home/amit/Data/OID/OID/Dataset/train/Person_Ambulance_Bus_Truck_Car_Tank_Taxi/Label"
    dest_path = source_path.replace("Label", "labels")

    os.makedirs(dest_path, exist_ok=False)

    all_label_files = os.listdir(source_path)

    cls_dict = {
                "Person": 0,
                "Ambulance": 1,
                "Bus": 1,
                "Truck": 1,
                "Car": 1,
                "Tank": 1,
                "Taxi": 1
                }

    pbar = tqdm(all_label_files, ncols=100)
    pbar.set_description('files converted')
    for label_file in pbar:
        if label_file.endswith('.txt'):
            with open(os.path.join(source_path, label_file), 'r') as label:
                img_file = os.path.join(source_path.replace("Label", "images"), label_file.replace(".txt", ".jpg"))
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
                    w = (xmax - xmin + 1) / W
                    h = (ymax - ymin + 1) / H

                    # # validating bbox size:
                    w = min(w, 2 * (1 - x0))
                    h = min(h, 2 * (1 - y0))

                    new_line = ' '.join([str(cls), str(x0), str(y0), str(w), str(h)])
                    bbox.append(new_line)

                # writing file to dest
                with open(dest_label_file, 'w') as dest_file:
                    for line in bbox:
                        dest_file.write("%s\n" % line)


