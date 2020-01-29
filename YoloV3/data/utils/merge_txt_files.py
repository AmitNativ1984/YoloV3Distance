import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

def concatenate_txt_files(txtfiles_list, merged_file_path):
    with open(merged_file_path, 'w') as merged_file:
        for txtfile in txtfiles_list:
            with open(txtfile) as f:
                for line in f:
                    merged_file.write(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create txt file from dataset")

    parser.add_argument('--txtfile', type=str, nargs='+',
                        required=True,
                        help='txt files to merge')

    parser.add_argument('--dest', type=str,
                        required=True,
                        help='path of merged txt file')


    args, unknow_args = parser.parse_known_args()

    concatenate_txt_files(args.txtfile, args.dest)