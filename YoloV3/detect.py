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
from YoloV3.utils.performence_measures import distance_over_IoU_thresh
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class Inference(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if args.cuda:
            self.model = self.model.cuda()

        # defining yolo detection layer:
        self.yolo_loss = YoloDetectionLayer(self.args)

        logging.info('Defining detector...')

        # Define dataloader
        kwargs = {'batch_size': 1, 'num_workers': args.workers, 'pin_memory': True}

        if self.args.dataset_type == 'distance':
            dataset = DistanceEstimationDataset(args, split='val')

        else:
            raise("unknown dataset type: {}".format(self.args.dataset_type))

        self.data_loader = torch.utils.data.DataLoader(dataset, **kwargs, shuffle=True)

        self.label_decoder = {v: k for k, v in dataset.label_decoding().items()}

    def infer(self, batch_inference=None):

        model = self.model
        model.eval()

        total_detction_time = 0.0
        num_frames = 0.
        GT_DIST = np.array([])
        PRED_DIST = np.array([])
        for b, sample in enumerate(self.data_loader):
            if batch_inference:
                if b >= batch_inference:
                    break


            image, bboxes = sample["image"], sample["bboxes"]

            if b == 0 or batch_inference is None:
                out_images = torch.zeros(1, image.shape[1], image.shape[2], image.shape[3])

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
                                                 scores=confident_bboxes[predicted_cls==curr_cls, 6 + curr_cls],
                                                 iou_threshold=self.args.nms_thres)

                        curr_bboxes = confident_bboxes[curr_detection_idx, :4]
                        curr_dist = confident_bboxes[curr_detection_idx, 5].view(-1, 1)
                        curr_cls = torch.Tensor(curr_detection_idx.shape[0], 1).cuda().fill_(curr_cls)

                        detections.append(torch.cat((curr_bboxes, curr_cls, curr_dist), dim=1))

                    detections = torch.cat(detections, dim=0).cpu().numpy()

                t_nms = time.time()

                # category_id_to_name = {0: 'Window', 1: 'door'}

                out_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

                pred_bboxes = detections[..., :4]
                pred_cls = detections[..., 4]
                pred_dist = detections[..., 5] * self.args.dist_norm

                try:
                    # if num_batches:
                    # adding groudtruth boxes
                    bboxes = bboxes[bboxes.sum(dim=-1) > 0].cpu().numpy()
                    bboxes[..., 4] = bboxes[..., 4] * self.args.dist_norm
                    gt_boxes = bboxes[..., :4]
                    gt_dist = bboxes[..., 4]# * self.args.dist_norm
                    gt_id = bboxes[..., 5]
                    annotation = {'image': out_image,
                                  'bboxes': gt_boxes,
                                  'category_id': gt_id,
                                  'dist': gt_dist}

                    out_image = visualize(annotation, self.label_decoder, Normalized=True, color=(1, 1, 0))

                    h, w, _ = out_image.shape

                    pred_bboxes[:, 0] /= w
                    pred_bboxes[:, 1] /= h
                    pred_bboxes[:, 2] /= w
                    pred_bboxes[:, 3] /= h

                    annotation = {'image': out_image,
                                  'bboxes': pred_bboxes,
                                  'category_id': pred_cls,
                                  'dist': pred_dist}

                    # detected image
                    out_image = visualize(annotation, self.label_decoder, Normalized=True, color=(1, 0, 0))
                    if batch_inference is None:
                        out_images = torch.tensor(out_image).permute(2, 1, 0).unsqueeze(0)

                    else:
                        out_images = torch.cat((out_images, torch.tensor(out_image).permute(2, 1, 0).unsqueeze(0)), dim=0)

                    predBOXES = np.concatenate([pred_bboxes, pred_dist.reshape(-1, 1), pred_cls.reshape(-1, 1)], axis=1)
                    gtBOXES = bboxes

                    # analyzing results
                    curr_gt_dist, curr_pred_dist = distance_over_IoU_thresh(gtBOXES, predBOXES, IoUth=0.95)

                    if curr_gt_dist.size > 0:
                        if GT_DIST.size == 0:
                            GT_DIST = curr_gt_dist
                        else:
                            GT_DIST = np.vstack((GT_DIST, curr_gt_dist))

                    if curr_pred_dist.size > 0:
                        if PRED_DIST.size == 0:
                            PRED_DIST = curr_pred_dist
                        else:
                            PRED_DIST = np.vstack((PRED_DIST, curr_pred_dist))

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

                cv2.namedWindow("output", cv2.WINDOW_NORMAL)

                cv2.imshow("output", cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
                cv2.resizeWindow("output", 512, 512)
                cv2.waitKey(1)
                # cv2.destroyAllWindows()
        cv2.destroyAllWindows()

        logging.info('average detection time: %.3f[msec]' % (total_detction_time / num_frames))

        # display distance stats:
        plt.figure()
        plt.title('distance estimation')
        plt.scatter(GT_DIST, PRED_DIST, marker='x', c='b')
        plt.xlabel('gt [m]')
        plt.ylabel('pred [m]')

        lr_model = np.polyfit(GT_DIST[:,0], PRED_DIST[:,0], 1) # perform linear regression
        dist_lr_pred = GT_DIST * lr_model[0] + lr_model[1]
        dist_rmse = np.sqrt(mean_squared_error(GT_DIST[:,0], PRED_DIST[:,0]))
        plt.plot(GT_DIST, dist_lr_pred, c='r')
        plt.text(4.5, 4.2, "y={:.5f}*x + {:.5f}\nRMSE={:.5f}".format(lr_model[0], lr_model[1], dist_rmse))
        plt.title('IoU=0.95')
        # plt.xlim((3, 7.5))
        # plt.ylim((3, 8))
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
    args.num_classes = len(DistanceEstimationDataset(args, split='val').label_decoding().keys())

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
    plt.show()

