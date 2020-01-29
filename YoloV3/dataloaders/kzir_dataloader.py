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

class KzirDataset(Dataset):
    """ Kzir dataset """

    def __init__(self, args, split="train"):
        self.split = split
        self.args = args

        if np.size(self.args.img_size) == 1:
            self.args.img_size = [self.args.img_size, self.args.img_size]

        self.cls_weights = self.class_weights()

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
        img = np.asarray(Image.open(img_path).convert('RGB')).astype(np.float32)

        # reading label
        bboxes = self.get_bbox_from_txt_file(label_path)

        if np.shape(bboxes)[0] > self.n_bbox:
            indices = np.random.choice(bboxes.shape[0], self.n_bbox, replace=False)
            bboxes = bboxes[indices]

        sample = {"image": img, "bboxes": bboxes}

        # # applying image transforms
        if self.split == "train":
            aug_sample = self.transform_tr(sample)
        elif self.split == "val" or self.split == "test":
            aug_sample = self.transform_val(sample)


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

        random_w = np.random.randint(img.shape[1] * 0.7, img.shape[1] * 0.95)
        random_h = np.random.randint(img.shape[0] * 0.7, img.shape[0] * 0.95)

        transforms = ([OneOf([Blur(p=0.5, blur_limit=(5, 10)),
                              MotionBlur(p=0.5, blur_limit=(5, 10))], p=0.2),
                      OneOf([HorizontalFlip(p=0.5),
                            RandomCrop(height=random_h, width=random_w, p=0.5),
                            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5)], p=0.5),
                      RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                      Resize(height=self.args.img_size[0], width=self.args.img_size[1], p=1),
                      Normalize(p=1.0)
                      ])

        preform_augmentation = Compose(transforms, bbox_params=BboxParams(format='yolo',
                                                                          min_area=0.0,
                                                                          min_visibility=0.0))
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

        transforms = ([
                       Resize(height=self.args.img_size[0], width=self.args.img_size[1], p=1),
                       Normalize(p=1, max_pixel_value=np.max(img))
                       ])

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
                if cls == 0 or cls==1 or cls == 3:    # no window
                    continue
                if cls==2:
                    cls = 0
                if cls==4:
                    cls = 1
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
        # labels = {
        #           "Window": 0,
        #           "Door": 1,
        #           "Person": 2,
        #           "House": 3,
        #           "Car": 4
        #           }

        labels = {
            "Person": 0,
            "Car": 1
        }

        return labels

    def class_weights(self):
        cls_weights = torch.tensor([1 / 2.137243975224324,  # Door
                                    1 / 65.41401190477923,  # Person
                                    1 / 10.8563089364414,  # House
                                    1 / 21.59243518355505])  # Car

        return cls_weights

if __name__ == "__main__":
    from DeepTools.argparse_utils.custom_arg_parser import CustomArgumentParser
    from DeepTools.data_augmentations.detection.custom_transforms import visualize
    parser = argparse.ArgumentParser(description="kzir database data loader", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = CustomArgumentParser(parser).convert_arg_line_to_args

    parser.add_argument('--database-root', type=str, default='/home/amit/Data/Kzir/Windows',
                       help='path to parent database root. its childern is images/ and /labels')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')

    parser.add_argument('--img-size', type=int, default=416,
                        help='image size')

    args, unknow_args = parser.parse_known_args()

    # creating train dataset
    train_set = KzirDataset(args, split='val')

    # creating train dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # looping over dataloader for sanity check
    pbar = tqdm.tqdm(train_loader)
    pbar.set_description("loaded data")
    category_id_to_name = {0: 'Window', 1: 'door'}
    for i, sample in enumerate(pbar):
        image, bbox = sample["image"], sample["bboxes"]
        annotation = {'image': sample["image"][0].permute(1, 2, 0).numpy(), 'bboxes': sample["bboxes"][0].numpy(), 'category_id': 0}
        visualize(annotation, category_id_to_name)

        time.sleep(0.05)



