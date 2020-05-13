import time
import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from YoloV3.dataloaders.distanceEstimation_dataloader import DistanceEstimationDataset
from YoloV3.nets.yolov3_tiny import YoloV3_tiny
from YoloV3.nets.yolov3 import YoloV3
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

        if self.args.dataset_type == 'distance':
            trainset = DistanceEstimationDataset(args, split='train')
            valset = DistanceEstimationDataset(args, split='val')

        else:
            raise("unknown dataset type: {}".format(self.args.dataset_type))

        self.train_loader = torch.utils.data.DataLoader(trainset, **kwargs, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(valset, **kwargs, shuffle=True)

        # defining yolo detection layer:
        self.yolo_loss = YoloDetectionLayer(self.args, cls_count=trainset.class_count())

        # Define optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        #                                  momentum=self.args.momentum)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(0.9*self.args.epochs), int(0.95*self.args.epochs)], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             factor=0.1,
        #                                                             patience=20,
        #                                                             verbose=True,
        #                                                             threshold=0.05,
        #                                                             threshold_mode='abs',
        #                                                             cooldown=0,
        #                                                             min_lr=1e-8,
        #                                                             eps=1e-08)

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

            # collecting loss from each layer:
            loss_total = torch.zeros([1], requires_grad=True).to(image.device)
            for pred in preds:

                # calculate loss for every resolution
                loss_out = self.yolo_loss(pred, bboxes)

                loss_total += loss_out[0]

                # accumulating loss for logging message:
                LossTotal += loss_out[0].item()
                BboxLoss += loss_out[1].item()
                ObjectnessLoss += loss_out[2].item()
                ClsLoss += loss_out[3].item()

            curr_time = time.time()
            pbar_postfix = '[loss: total=%.5f| bbox=%.5f| objctness=%.5f| cls=%.5f] lr = %.6f time=%.2f [min]' % (LossTotal/batch,
                                                                                                   BboxLoss/batch,
                                                                                                   ObjectnessLoss/batch,
                                                                                                   ClsLoss/batch,
                                                                                                   self.optimizer.param_groups[0]['lr'],
                                                                                                   (curr_time - start_time)/60)
            pbar.set_postfix_str(s=pbar_postfix, refresh=True)

            # backpropagation
            loss_total.backward()
            self.optimizer.step()

        # writing epoch summaries to tensorboard:
        self.writer.add_scalar('train/total_loss_epoch', LossTotal/batch, epoch)
        self.writer.add_scalar('train/BboxLoss', BboxLoss/batch, epoch)
        self.writer.add_scalar('train/objectness_loss', ObjectnessLoss/batch, epoch)
        self.writer.add_scalar('train/cls_loss', ClsLoss/batch, epoch)

        #self.scheduler.step(LossTotal/batch)
        self.scheduler.step()

    def validate(self, epoch):

        model = self.model
        model.eval()

        model_was_saved = False

        LossTotal = 0.
        BboxLoss = 0.
        ObjectnessLoss = 0.
        ClsLoss = 0.

        barformat = "{l_bar}{bar}|{n_fmt}/{total_fmt}{postfix}"
        pbar = tqdm(self.val_loader, ncols=150, bar_format=barformat)
        pbar.set_description('Val   [Epoch %d/%d]' % (epoch + 1, self.args.epochs))
        start_time = time.time()
        for b, sample in enumerate(pbar):
            batch = b + 1
            image, bboxes = sample["image"], sample["bboxes"]
            if self.args.cuda:
                image, bboxes = image.to(device), bboxes.to(device)

            with torch.no_grad():
                preds = model(image)

                # collecting loss from each layer:
                loss_total = torch.zeros([1], requires_grad=True).to(image.device)
                for pred in preds:
                    # calculate loss for every resolution
                    loss_out = self.yolo_loss(pred, bboxes)

                    loss_total += loss_out[0]

                    # accumulating loss for logging message:
                    LossTotal += loss_out[0].item()
                    BboxLoss += loss_out[1].item()
                    ObjectnessLoss += loss_out[2].item()
                    ClsLoss += loss_out[3].item()

                    curr_time = time.time()

            pbar_postfix = '[loss: total=%.5f| bbox=%.5f| objctness=%.5f| cls=%.5f] time=%.2f [min]' % (
                                                                                                        LossTotal / batch,
                                                                                                        BboxLoss / batch,
                                                                                                        ObjectnessLoss / batch,
                                                                                                        ClsLoss / batch,
                                                                                                        (curr_time - start_time) / 60)
            pbar.set_postfix_str(s=pbar_postfix, refresh=True)

        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # saving best results.
        if LossTotal / batch < self.min_loss:
            logging.info('validation loss imporoved from {} to {}'.format(self.min_loss, LossTotal/batch))
            self.min_loss = LossTotal/batch

            model_outpath = os.path.join(self.args.output_path, self.args.checkpoint)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.min_loss
                        }, model_outpath)

            logging.info('Model saved to: {}'.format(model_outpath))

        # saving model to save file on every epoch
        model_outpath = os.path.join(self.args.output_path, 'last_trained_epoch.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.min_loss
        }, model_outpath)

        # writing epoch summaries to tensorboard:
        self.writer.add_scalar('val/total_loss_epoch', LossTotal / batch, epoch)
        self.writer.add_scalar('val/BboxLoss', BboxLoss / batch, epoch)
        self.writer.add_scalar('val/objectness_loss', ObjectnessLoss / batch, epoch)
        self.writer.add_scalar('val/cls_loss', ClsLoss / batch, epoch)

        # running inference and saving to tensorboard
        infer_args = self.args
        infer_args.batch_size = 1
        out_images = Inference(infer_args, model).infer(num_batches=5)
        self.writer.add_image('detections', out_images, global_step=epoch, dataformats='NCWH')

        # torch.cuda.empty_cache()
        # logging.info('emptied cuda cache successfully')
        logging.info('\n')

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
                        choices=['yolov3-tiny', 'yolov3'],
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
    parser.add_argument('--weights', type=str, default='',
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
                        default='distance',
                        help='dataset type')
    # checking point
    parser.add_argument('--resume', action='store_true',
                        help='is given, will resu,e training from loaded checkpoint including learning rate')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='set the checkpoint name')
    parser.add_argument('--output-path', type=str, default=os.getcwd() + "/YoloV3/results",
                        help='output path')

    # detection parameters
    parser.add_argument('--nms-thres', type=float, default=0.1,
                        help='non max suppression bbox iou threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='object prediction confidence threshold')
    parser.add_argument('--dist-norm', type=float, default=30.0,
                        help='normalize distance by this size')

    args, unknow_args = parser.parse_known_args()

    logging.info('input parameters: {}'.format(args))

    # getting number of classes from dataloader
    args.num_classes = len(DistanceEstimationDataset(args).label_decoding().keys())

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
    elif args.model == 'yolov3':
        model = YoloV3(args)
    else:
        raise ("currently supporting only yolov3_or yolov3 tiny")


    if args.weights != '':
        # load trained model
        model.load_state_dict(torch.load(args.weights)["model_state_dict"])
        logging.info('loaded model from: {}'.format(args.weights))


    # Multi GPU if possible:
    if args.cuda and len(args.gpu_ids) > 1:
        logging.info("Using {} GPUs!".format(len(args.gpu_ids)))
        model = torch.nn.DataParallel(model)
        model.to(device)


    trainer = Trainer(args, model)

    if args.resume:
        trainer.optimizer.load_state_dict(torch.load(args.weights)["optimizer"])

    # BEGIN TRAINING
    # --------------
    logging.info('Begin training...')
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.validate(epoch)

    logging.info("training finished")


