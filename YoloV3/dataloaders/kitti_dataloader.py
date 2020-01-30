import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tqdm
import random
import matplotlib.pyplot as plt
import time
import cv2

from albumentations import (BboxParams,
                            HorizontalFlip,
                            Resize,
                            RandomCrop,
                            RandomScale,
                            PadIfNeeded,
                            ShiftScaleRotate,
                            Blur,
                            MotionBlur,
                            Normalize,
                            RandomBrightnessContrast,
                            OneOf,
                            Compose)

from DeepTools.data_augmentations.detection import custom_transforms as tr

class KittiDataset(Dataset):

    """ Kitti dataset """

    def __init__(self, args, split="train"):
        self.split = split
        self.args = args

        if np.size(self.args.img_size) == 1:
            self.args.img_size = [self.args.img_size, self.args.img_size]

        if self.split == 'train':
            files_list = self.args.train_data
        elif self.split == 'val':
            files_list = self.args.val_data

        self.n_bbox = 50

        self.files = []
        with open(files_list, "r") as f:
            for line in f:
                self.files.append(line.strip())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        _, file_ext = os.path.splitext(img_path)
        label_path = img_path.replace("images", "labels").replace(file_ext, ".txt")

        # reading image:
        img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.uint8)

        # reading label
        bboxes = self.get_bbox_from_txt_file(label_path)

        if np.shape(bboxes)[0] > self.n_bbox:
            indices = np.random.choice(bboxes.shape[0], self.n_bbox, replace=False)
            bboxes = bboxes[indices]

        sample = {"image": img, "bboxes": bboxes}

        num_augmented_bbox = 0
        counter = 0
        while counter < 10 and num_augmented_bbox == 0:
            # # applying image transforms
            if self.split == "train":
                aug_sample = self.transform_tr(sample)
            elif self.split == "val" or self.split == "test":
                aug_sample = self.transform_val(sample)

            counter += 1
            num_augmented_bbox = aug_sample["bboxes"].size


        # creating fixed size bbox tensor

        bboxes = -1 * np.ones((self.n_bbox, 5))
        N_bbox = np.shape(sample["bboxes"])[0]
        N_augmented_bbox = np.shape(aug_sample["bboxes"])[0]
        if N_augmented_bbox > 0 and N_bbox > 0:
            bboxes[:N_augmented_bbox, :] = aug_sample["bboxes"]

        aug_sample["bboxes"] = bboxes

        # creating pytorch tensors:
        aug_sample["image"] = torch.tensor(aug_sample["image"], dtype=torch.float).permute(2, 0, 1)
        aug_sample["bboxes"] = torch.tensor(aug_sample["bboxes"], dtype=torch.float)

        return aug_sample

    def transform_tr(self, sample):
        """ data augmentation for training """
        img = sample["image"]
        bboxes = sample["bboxes"]

        if bboxes.size == 0:
            bboxes = np.array([[0.1, 0.1, 0.1, 0.1, 0.0]])  # this is just a dummy - all values must be inside (0,1)

        annotations = {'image': img, 'bboxes': bboxes}

        random_scale = np.random.randint(8, 11)/10

        transforms = ([OneOf([Blur(p=0.5, blur_limit=(3, 10)),
                              MotionBlur(p=0.5, blur_limit=(3, 20))], p=0.2),
                      # padding image in case is too small
                      PadIfNeeded(min_height=self.args.img_size[0], min_width=self.args.img_size[1],
                                  border_mode=cv2.BORDER_REPLICATE,
                                  p=1.0),
                      # changing image size - mainting aspect ratio for later resize
                      OneOf([RandomCrop(height=self.args.img_size[0], width=self.args.img_size[1], p=0.5),
                              RandomCrop(height=int(random_scale * self.args.img_size[0]),
                                         width=int(random_scale * self.args.img_size[1]), p=0.5)], p=1.0),
                      # flipping / rotating
                      OneOf([HorizontalFlip(p=0.5),
                            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5)], p=0.5),
                      # contrast and brightness
                      RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                      # making sure resize fits with yolo input size
                      Resize(height=self.args.img_size[0], width=self.args.img_size[1], p=1.0),
                      Normalize(p=1.0)])

        preform_augmentation = Compose(transforms, bbox_params=BboxParams(format='yolo',
                                                                          min_visibility=0.2))
        augmented_sample = preform_augmentation(**annotations)

        augmented_sample["bboxes"] = np.array(augmented_sample["bboxes"])

        return augmented_sample

    def transform_val(self, sample):
        """ data augmentation for training """
        img = sample["image"]
        bboxes = sample["bboxes"]

        if bboxes.size == 0:
            bboxes = np.array([[0.1, 0.1, 0.1, 0.1, 0.0]])  # this is just a dummy - all values must be inside (0,1)

        annotations = {'image': img, 'bboxes': bboxes}

        transforms = ([PadIfNeeded(min_height=self.args.img_size[0], min_width=self.args.img_size[1],
                                   border_mode=cv2.BORDER_REPLICATE, p=1.0),
                       # changing image size - mainting aspect ratio for later resize
                       RandomCrop(height=self.args.img_size[0], width=self.args.img_size[1], p=1.0),
                       Normalize(p=1.0)])

        preform_augmentation = Compose(transforms, bbox_params=BboxParams(format='yolo',
                                                                          min_area=0.0,
                                                                          min_visibility=0.0))
        augmented_sample = preform_augmentation(**annotations)

        augmented_sample["bboxes"] = np.array(augmented_sample["bboxes"])

        return augmented_sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """

        valid_image_files = []
        for looproot, _, filenames in os.walk(rootdir):
            for filename in filenames:
                if filename.endswith(suffix):
                    image_path = os.path.join(looproot, filename)
                    label_path = image_path.replace("images", "labels").replace("bmp", "txt")
                    if os.path.isfile(label_path):
                       valid_image_files.append(image_path)

        return valid_image_files

    def get_bbox_from_txt_file(self, txt_file):
        """
        return bounding box numpy array in YOLO format:
        [x0, y0, w, h, cls]
        x0, y0, w, h: are normalized by image size and lie between (0,1)
        cls is the the cls index
        """
        bbox = []
        with open(txt_file, 'r') as f:
            for line in f:
                vec = line.split(' ')
                cls = int(vec[0])
                if cls != 0 and cls != 1:
                    raise('wrong cls id in file: {}'.format(txt_file))

                x0 = float(vec[1])
                y0 = float(vec[2])
                w = float(vec[3])
                h = float(vec[4])

                # validating bbox size:
                w = min(w, 2 * (1 - x0 - 1e-10), 2 * x0)
                h = min(h, 2 * (1 - y0 - 1e-10), 2 * y0)

                bbox.append([x0, y0, w, h, cls])

        bbox = np.array(bbox)
        return bbox

    def label_decoding(self):

        labels = {
            "Person": 0,
            "Car": 1
        }

        return labels

    def class_count(self):
        from ..data.utils.count_cls import count_cls as countCls

        cls_indx = list(self.label_decoding().values())
        cls_indx = [str(i) for i in cls_indx]
        cls_names = list(self.label_decoding().keys())
        cls_dict = dict(zip(cls_indx, cls_names))

        cls_count_dict = countCls(self.args.train_data, cls_dict)

        cls_count = [None] * len(cls_names)
        # turning dict to array in correct order of classes indx:
        for cls_name in cls_names:
            cls_count[self.label_decoding()[cls_name]] = cls_count_dict[cls_name]

        print("\ncls count: {}\n".format(cls_count))

        return np.array(cls_count)

if __name__ == "__main__":
    from DeepTools.argparse_utils.custom_arg_parser import CustomArgumentParser
    from DeepTools.data_augmentations.detection.custom_transforms import visualize
    parser = argparse.ArgumentParser(description="kitti database data loader", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = CustomArgumentParser(parser).convert_arg_line_to_args

    parser.add_argument('--train-data', type=str,
                        required=True,
                        help='txt file containt all train images path')

    parser.add_argument('--val-data', type=str,
                        required=True,
                        help='txt file containt all train images path')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')

    parser.add_argument('--img-size', type=int, default=416,
                        help='image size')

    args, unknow_args = parser.parse_known_args()

    # creating train dataset
    train_set = KittiDataset(args, split='train')

    # creating train dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # looping over dataloader for sanity check
    pbar = tqdm.tqdm(train_loader)
    pbar.set_description("loaded data")
    category_id_to_name = {0: 'Person', 1: 'Car'}
    for i, sample in enumerate(pbar):
        image, bbox = sample["image"], sample["bboxes"]

        annotation = {'image': image[0].permute(1, 2, 0).numpy(), 'bboxes':bbox[0].numpy(), 'category_id': bbox[0][bbox[0,:,-1]>-1][:,-1]}
        visualize(annotation, category_id_to_name)

        time.sleep(0.05)



