import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from YoloV3.dataloaders.distanceEstimation_dataloader import DistanceEstimationDataset
from YoloV3.nets.yolov3_tiny import YoloV3_tiny
from YoloV3.nets.yolov3 import YoloV3
from YoloV3.nets.yolo_basic_blocks import YoloDetectionLayer
import torchvision
from torchvision.ops.boxes import nms, box_iou
from DeepTools.argparse_utils.custom_arg_parser import CustomArgumentParser
import logging
from DeepTools.tensorboard_writer.summaries import TensorboardSummary
import cv2
import time
from DeepTools.data_augmentations.detection.custom_transforms import visualize

class Inference(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if args.cuda:
            self.model = self.model.cuda()

        # defining yolo detection layer:
        self.yolo_loss = YoloDetectionLayer(self.args)

        logging.info('Defining trainer...')

        # Define dataloader
        kwargs = {'batch_size': 1, 'num_workers': args.workers, 'pin_memory': True}

        if self.args.dataset_type == 'distance':
            dataset = DistanceEstimationDataset(args, split='val')

        else:
            raise("unknown dataset type: {}".format(self.args.dataset_type))

        self.data_loader = torch.utils.data.DataLoader(dataset, **kwargs, shuffle=True)

        self.label_decoder = {v: k for k, v in dataset.label_decoding().items()}

    def infer(self, num_batches=None):

        model = self.model
        model.eval()

        total_detction_time = 0.0
        num_frames = 0.
        out_images = []
        for b, sample in enumerate(self.data_loader):
            if num_batches:
                if b >= num_batches:
                    break


            image, bboxes = sample["image"], sample["bboxes"]

            if b == 0 and num_batches:
                out_images = torch.zeros(num_batches, image.shape[1], image.shape[2], image.shape[3])

            if self.args.cuda:
                image, bboxes = image.cuda(), bboxes.cuda()

            t0 = time.time()
            with torch.no_grad():
                preds = model(image)
                num_frames += 1

                for res_idx, pred in enumerate(preds):
                    # calculate loss for every resolution
                    curr_res_output = self.yolo_loss(pred)
                    if res_idx == 0:
                        output = curr_res_output
                    else:
                        output = torch.cat((output, curr_res_output), dim=1)

                t_preds = time.time()

                # discard all bboxes with conf < threshold:
                confident_bboxes = output[output[..., 4] > self.args.conf_thres, :]
                if confident_bboxes.numel() == 0:
                    detections = confident_bboxes # empty tensor

                else:
                    predicted_cls = torch.argmax(confident_bboxes[..., 6:], dim=-1).cpu().numpy()

                    # finding all possible classes in image
                    all_cls_in_image = np.unique(predicted_cls)

                    detections = []
                    # performing nms for each class separately.
                    # all detections will be appended and later concatenated to single tensor
                    for curr_cls in all_cls_in_image:
                        boxes = confident_bboxes[predicted_cls==curr_cls, :4]
                        w = boxes[:, 2]
                        h = boxes[:, 3]
                        x1 = boxes[:, 0] - w / 2
                        y1 = boxes[:, 1] - h / 2
                        x2 = boxes[:, 0] + w / 2
                        y2 = boxes[:, 1] + h / 2
                        boxes[:, 0] = x1
                        boxes[:, 1] = y1
                        boxes[:, 2] = x2
                        boxes[:, 3] = y2

                        curr_detection_idx = nms(boxes=boxes,
                                                 scores=confident_bboxes[predicted_cls==curr_cls, 5 + curr_cls],
                                                 iou_threshold=self.args.nms_thres)

                        curr_bboxes = confident_bboxes[curr_detection_idx, :4]
                        curr_dist = confident_bboxes[curr_detection_idx, 5]
                        curr_cls = torch.Tensor(curr_detection_idx.shape[0], 1).cuda().fill_(curr_cls)

                        detections.append(torch.cat((curr_bboxes, curr_cls), dim=1))

                    detections = torch.cat(detections, dim=0).cpu().numpy()

                t_nms = time.time()

                # category_id_to_name = {0: 'Window', 1: 'door'}


                annotation = {'image': image.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                              'bboxes': detections[..., :4],
                              'category_id': detections[..., 4]}

                try:
                    out_image = visualize(annotation, self.label_decoder, Normalized=False, color=(0, 0, 1))


                    if num_batches:
                        # adding groudtruth boxes
                        gt_boxes = bboxes[bboxes.sum(dim=-1) > 0][..., :4]
                        gt_id = bboxes[bboxes.sum(dim=-1) > 0][..., 4].cpu().numpy()

                        annotation = {'image': out_image,
                                      'bboxes': gt_boxes,
                                      'category_id': gt_id}

                        out_image = visualize(annotation, self.label_decoder, Normalized=True, color=(0, 1, 0))


                        out_images[b, ...] = torch.tensor(out_image).permute(2, 1, 0)

                except Exception as e:
                    print(e)

                t_visualize = time.time()

                logging.info("#detections: %d total time:%.3f[msec], pred: %.3f[msec], display: %.3f[msec]" % (
                                                                                                     detections.shape[0],
                                                                                                     (t_visualize-t0)*1e3,
                                                                                                     (t_preds - t0)*1e3 +
                                                                                                     (t_nms - t_preds) * 1e3,
                                                                                                     (t_visualize - t_nms) * 1e3))


                total_detction_time += (t_preds - t0)*1e3 + (t_nms - t_preds) * 1e3
        cv2.destroyAllWindows()

        logging.info('average detection time: %.3f[msec]' % (total_detction_time / num_frames))
        return out_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kzir database data loader", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = CustomArgumentParser(parser).convert_arg_line_to_args

    parser.add_argument('--val-data', type=str,
                        required=True,
                        help='path to parent database root. its childern is images/ and /labels')
    parser.add_argument('--model', type=str, default='yolov3-tiny',
                        choices=['yolov3-tiny', 'yolov3'],
                        help='yolo models. can be one of: yolov3-tiny, yolov3')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--img-size', type=int, default=416,
                        help='image size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to pretrained weights')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                               comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--nms-thres', type=float, default=0.1,
                        help='non max suppression bbox iou threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                        help='object prediction confidence threshold')


    # classes and anchors:
    parser.add_argument('--num-classes', type=int, default=5,
                        help="number of classes")
    parser.add_argument('--anchors', type=str,
                        default='[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]',
                        help='lists of anchors. each anchor are given as lists separated by coma: [x1,y1],[x2,y2],..')
    parser.add_argument('--num-anchors-per-resolution', type=int,
                        default=3,
                        help='lists of anchors. each anchor are given as lists separated by coma: [x1,y1],[x2,y2],..')
    parser.add_argument('--dataset-type', type=str,
                        default='kzir',
                        help='dataset type. one of kitti, kzir')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--output-path', type=str, default=os.getcwd() + "/YoloV3/results/model_best.pth",
                        help='output path')


    args, unknow_args = parser.parse_known_args()

    logging.info('input parameters: {}'.format(args))

    # parsing anchors:
    anchors = args.anchors.replace('[',',').replace(']',',').split(',')
    anchors = np.array([int(num_str) for num_str in anchors if num_str])
    args.anchors = np.reshape(anchors, (-1, 2))

    # setting image size:
    if np.size(args.img_size) == 1:
        args.img_size = [args.img_size, args.img_size]

        # selecting model from user inputs
        if args.model == 'yolov3-tiny':
            model = YoloV3_tiny(args)
        elif args.model == 'yolov3':
            model = YoloV3(args)
        else:
            raise ("currently supporting only yolov3_or yolov3 tiny")

    args.cuda = torch.cuda.is_available()
    if args.cuda:
        try:
            torch.cuda.empty_cache()
            logging.info('emptied cuda cache successfully')
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            if len(args.gpu_ids) == 1 and args.gpu_ids[0] != torch.cuda.current_device():
                torch.cuda.set_device(args.gpu_ids[0])
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    # load trained model
    model.load_state_dict(torch.load(args.output_path)["model_state_dict"])
    logging.info('loaded model from: {}'.format(args.output_path))


    logging.info('run inference...')
    inference = Inference(args, model).infer()

    logging.info("inference finished")


