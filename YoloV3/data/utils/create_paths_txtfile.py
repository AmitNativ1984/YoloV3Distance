import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create txt file from dataset")

    parser.add_argument('--images-path', type=str,
                        required=True,
                        help='path where images are stored. must be under ./images/ labels under ./labels/')

    parser.add_argument('--dest-txtfile', type=str,
                        required=True,
                        help='txt file containt all train images path')

    parser.add_argument('--txt-file-mode', type=str, default='a+',
                        help='txt file writing mode: a, a+, w, w+, ...')


    args, unknow_args = parser.parse_known_args()

    all_images_path = os.listdir(args.images_path)

    with open(args.dest_txtfile, args.txt_file_mode) as dest_file:
        for image_name in tqdm(all_images_path, desc='writing paths'):
            img_path = os.path.join(args.images_path, image_name)
            label_path = img_path.replace("/images", "/labels").replace(os.path.splitext(img_path)[-1], ".txt")
            if os.path.isfile(label_path):
                if os.path.getsize(label_path):
                    dest_file.write("%s\n" % img_path)
                else:
                    print('empty labels')

    # txtfiles_list = [os.path.join(os.getcwd(), 'YoloV3/data/OID_train.txt'),
    #                  os.path.join(os.getcwd(), 'YoloV3/data/kzir_train.txt')]
    #
    # merged_file = os.path.join(os.getcwd(), 'YoloV3/data/OID_kzir_merged_train.txt')
    # concatenate_txt_files(txtfiles_list, merged_file)

