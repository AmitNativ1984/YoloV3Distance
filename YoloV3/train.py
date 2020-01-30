import time
import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from YoloV3.dataloaders.kzir_dataloader import KzirDataset
from YoloV3.dataloaders.kitti_dataloader import KittiDataset
from YoloV3.nets.yolov3_tiny import YoloV3_tiny
from YoloV3.nets.yolo_basic_blocks import YoloDetectionLayer
from YoloV3.detect import Inference
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from DeepTools.argparse_utils.custom_arg_parser import CustomArgumentParser
from DeepTools.data_augmentations.detection import custom_transforms as tr
import logging
from DeepTools.tensorboard_writer.summaries import TensorboardSummary
from datetime import datetime

class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

        # define tensorboard summary
        time_stamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        tb_log_path = os.path.join(self.args.output_path, 'logs', 'lr_' + str(self.args.lr) + '_'+time_stamp)
        self.writer = TensorboardSummary(tb_log_path).create_summary()

        if args.cuda:
            self.model = self.model.cuda()

        logging.info('Defining trainer...')

        # Define dataloader
        kwargs = {'batch_size': args.batch_size, 'num_workers': args.workers, 'pin_memory': True}

        if self.args.dataset_type == 'kzir':
            trainset = KzirDataset(args, split='train')
            valset = KzirDataset(args, split='val')

        elif self.args.dataset_type == 'kitti':
            trainset = KittiDataset(args, split='train')
            valset = KittiDataset(args, split='val')
        else:
            raise("unknown dataset type: {}".format(self.args.dataset_type))

        self.train_loader = torch.utils.data.DataLoader(trainset, **kwargs, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valset, **kwargs, shuffle=False)

        # defining yolo detection layer:
        self.yolo_loss = YoloDetectionLayer(self.args, cls_count=trainset.class_count())

        # Define optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                         momentum=self.args.momentum)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[300, 400], gamma=0.1)

        self.min_loss = np.inf

    def train(self, epoch):
        """ runs training for current epoch """

        model = self.model
        model.train()

        LossTotal = 0.
        BboxLoss = 0.
        ObjectnessLoss = 0.
        ClsLoss = 0.

        barformat = "{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}"
        pbar = tqdm(self.train_loader, ncols=150, bar_format=barformat)
        pbar.set_description('Train [Epoch %d/%d]' % (epoch + 1, self.args.epochs))
        start_time = time.time()
        for b, sample in enumerate(pbar):
            batch = b + 1
            image, bboxes = sample["image"], sample["bboxes"]
            if self.args.cuda:
                image, bboxes = image.to(device), bboxes.to(device)


            self.optimizer.zero_grad()
            preds = model(image)

            # calculate loss for every resolution
            loss_out0 = self.yolo_loss(preds[0], bboxes)
            loss_out1 = self.yolo_loss(preds[1], bboxes)

            loss_total = loss_out0[0] + loss_out1[0]
            bbox_loss = loss_out0[1] + loss_out1[1]
            objectness_loss = loss_out0[2] + loss_out1[2]
            cls_loss = loss_out0[3] + loss_out1[3]

            LossTotal += loss_total.item()
            BboxLoss += bbox_loss.item()
            ObjectnessLoss += objectness_loss.item()
            ClsLoss += cls_loss.item()

            curr_time = time.time()
            pbar_postfix = '[loss: total=%.5f| bbox=%.5f| objctness=%.5f| cls=%.5f] time=%.2f [sec]' % (LossTotal/batch,
                                                                                                   BboxLoss/batch,
                                                                                                   ObjectnessLoss/batch,
                                                                                                   ClsLoss/batch,
                                                                                                   curr_time - start_time)
            pbar.set_postfix_str(s=pbar_postfix, refresh=True)

            # backpropagation
            loss_total.backward()
            self.optimizer.step()

        # writing epoch summaries to tensorboard:
        self.writer.add_scalar('train/total_loss_epoch', LossTotal/batch, epoch)
        self.writer.add_scalar('train/BboxLoss', BboxLoss/batch, epoch)
        self.writer.add_scalar('train/objectness_loss', ObjectnessLoss/batch, epoch)
        self.writer.add_scalar('train/cls_loss', ClsLoss/batch, epoch)

        self.scheduler.step(epoch)

    def validate(self, epoch):

        model = self.model
        model.eval()

        model_was_saved = False

        loss_total = 0.
        bbox_loss = 0.
        objectness_loss = 0.
        cls_loss = 0.

        barformat = "{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}"
        pbar = tqdm(self.val_loader, ncols=150, bar_format=barformat)
        pbar.set_description('Val   [Epoch %d/%d]' % (epoch + 1, self.args.epochs))
        start_time = time.time()
        for b, sample in enumerate(pbar):
            batch = b + 1
            image, bboxes = sample["image"], sample["bboxes"]
            if self.args.cuda:
                image, bboxes = image.cuda(), bboxes.cuda()

            with torch.no_grad():
                preds = model(image)

                # calculate loss for every resolution
                loss_out0 = self.yolo_loss(preds[0], bboxes)
                loss_out1 = self.yolo_loss(preds[1], bboxes)

                loss_total += loss_out0[0] + loss_out1[0].item()
                bbox_loss += loss_out0[1] + loss_out1[1].item()
                objectness_loss += loss_out0[2] + loss_out1[2]
                cls_loss += loss_out0[3] + loss_out1[3]

                curr_time = time.time()
                pbar_postfix = '[loss: total=%.5f| bbox=%.5f| objctness=%.5f| cls=%.5f] time=%.2f [sec]' % (loss_total.item()/batch,
                                                                                                            bbox_loss.item()/batch,
                                                                                                            objectness_loss.item()/batch,
                                                                                                            cls_loss.item()/batch,
                                                                                                            curr_time - start_time)
                pbar.set_postfix_str(s=pbar_postfix, refresh=True)

        epoch_total_loss = loss_total.item() / batch

        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # saving model to save file on every epoch
        model_outpath = os.path.join(self.args.output_path, 'last_trained_epoch.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.min_loss
        }, model_outpath)

        # saving best results.
        if epoch_total_loss < self.min_loss:
            self.min_loss = epoch_total_loss

            model_outpath = os.path.join(self.args.output_path, self.args.checkpoint)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.min_loss
                        }, model_outpath)

        # writing epoch summaries to tensorboard:
        self.writer.add_scalar('val/total_loss_epoch', loss_total.item()/batch, epoch)
        self.writer.add_scalar('val/BboxLoss', bbox_loss.item()/batch, epoch)
        self.writer.add_scalar('val/objectness_loss', objectness_loss.item()/batch, epoch)
        self.writer.add_scalar('val/cls_loss', cls_loss.item()/batch, epoch)

        # running inference and saving to tensorboard
        infer_args = self.args
        infer_args.batch_size = 1
        out_images = Inference(infer_args, model).infer(num_batches=5)
        self.writer.add_image('detections', out_images, global_step=epoch, dataformats='NCWH')

        torch.cuda.empty_cache()
        logging.info('emptied cuda cache successfully')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kzir database data loader", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = CustomArgumentParser(parser).convert_arg_line_to_args

    parser.add_argument('--train-data', type=str,
                        required=True,
                        help='txt file containt all train images path')
    parser.add_argument('--val-data', type=str,
                        required=True,
                        help = 'path to parent database root. its childern is images/ and /labels')

    parser.add_argument('--model', type=str, default='yolov3-tiny',
                        choices=['yolov3-tiny'],
                        help='yolo models. can be one of: yolov3-tiny, yolov3')
    parser.add_argument('--batch-size', type=int, default=14,
                        help='train batch size')
    parser.add_argument('--img-size', type=int, default=416,
                        help='image size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to pretrained weights')
    parser.add_argument('--gpu-ids', type=str, default='1, 0',
                        help='use which gpu to train, must be a \
                               comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')

    # classes and anchors:
    # parser.add_argument('--num-classes', type=int, default=KzirDataset().label_decoding,
    #                     help="number of classes")
    parser.add_argument('--anchors', type=str,
                        default='[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]',
                        help='lists of anchors. each anchor are given as lists separated by coma: [x1,y1],[x2,y2],..')
    parser.add_argument('--num-anchors-per-resolution', type=int,
                        default=3,
                        help='lists of anchors. each anchor are given as lists separated by coma: [x1,y1],[x2,y2],..')
    # dataset type
    parser.add_argument('--dataset-type', type=str,
                        default='kzir',
                        help='dataset type. one of kitti, kzir')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='set the checkpoint name')
    parser.add_argument('--output-path', type=str, default=os.getcwd() + "/YoloV3/results",
                        help='output path')

    # detection parameters
    parser.add_argument('--nms-thres', type=float, default=0.1,
                        help='non max suppression bbox iou threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='object prediction confidence threshold')

    args, unknow_args = parser.parse_known_args()

    logging.info('input parameters: {}'.format(args))

    # getting number of classes from dataloader
    args.num_classes = len(KzirDataset(args).label_decoding().keys())

    # parsing anchors:
    anchors = args.anchors.replace('[',',').replace(']',',').split(',')
    anchors = np.array([int(num_str) for num_str in anchors if num_str])
    args.anchors = np.reshape(anchors, (-1, 2))

    # setting image size:
    if np.size(args.img_size) == 1:
        args.img_size = [args.img_size, args.img_size]

    # setting output path
    os.makedirs(args.output_path, exist_ok=True)

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            torch.cuda.empty_cache()
            logging.info('emptied cuda cache successfully')
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            if len(args.gpu_ids) == 1 and args.gpu_ids[0] != torch.cuda.current_device():
                torch.cuda.set_device(args.gpu_ids[0])

            device = torch.cuda.current_device()

        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    # selecting model from user inputs
    if args.model == 'yolov3-tiny':
        model = YoloV3_tiny(args)
    else:
        raise ("currently supporting only yolov3_tiny")


    if args.resume:
        # load trained model
        model.load_state_dict(torch.load(args.resume)["model_state_dict"])
        logging.info('loaded model from: {}'.format(args.output_path))


    # Multi GPU if possible:
    if args.cuda and len(args.gpu_ids) > 1:
        logging.info("Using {} GPUs!".format(len(args.gpu_ids)))
        model = torch.nn.DataParallel(model)
        model.to(device)

    trainer = Trainer(args, model)

    if args.resume:
        trainer.optimizer.load_state_dict(torch.load(args.resume)["optimizer"])

    # BEGIN TRAINING
    # --------------
    logging.info('Begin training...')
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.validate(epoch)

    logging.info("training finished")


