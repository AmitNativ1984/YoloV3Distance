"""
this file contains yolo layers that are common in any backbone (such as darknet-53, yolov3-tiny, etc...)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from DeepTools.utils.object_detection_utils import bbox_iou

class conv2D_BN_LeakyRelu(nn.Module):
    def __init__(self, inplanes, outplanes, kernelsize, stride, padding):
        super(conv2D_BN_LeakyRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                               kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        self.batchNorm = nn.BatchNorm2d(num_features=outplanes)
        self.LeakyRelu = nn.LeakyReLU(0.1)

        self.conv2dBNLeakyRelu = nn.Sequential(self.conv2d,
                                               self.batchNorm,
                                               self.LeakyRelu)


    def forward(self, x):

        x = self.conv2dBNLeakyRelu(x)
        return x

class MaxPoolPaddedStride1(nn.Module):
    def __init__(self):
        super(MaxPoolPaddedStride1, self).__init__()

    def forward(self, x):

        x = F.pad(x, (0, 1, 0, 1), mode="replicate")
        x = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()

        self.conv1 = conv2D_BN_LeakyRelu(inplanes=inplanes, outplanes=planes[0], kernelsize=1, stride=1, padding=0)
        self.conv2 = conv2D_BN_LeakyRelu(inplanes=planes[0], outplanes=planes[1], kernelsize=3, stride=1, padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual

        return out

class ConvSet(nn.Module):
    def __init__(self, inplanes, planes):
        super(ConvSet, self).__init__()

        self.conv0 = conv2D_BN_LeakyRelu(inplanes=inplanes, outplanes=planes[0], kernelsize=1, stride=1, padding=0)
        self.conv1 = conv2D_BN_LeakyRelu(inplanes=planes[0], outplanes=planes[1], kernelsize=3, stride=1, padding=1)
        self.conv2 = conv2D_BN_LeakyRelu(inplanes=planes[1], outplanes=planes[0], kernelsize=1, stride=1, padding=0)
        self.conv3 = conv2D_BN_LeakyRelu(inplanes=planes[0], outplanes=planes[1], kernelsize=3, stride=1, padding=1)
        self.conv4 = conv2D_BN_LeakyRelu(inplanes=planes[1], outplanes=planes[0], kernelsize=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.conv4(x)

        return out

class YoloDetectionLayer(nn.Module):
    """ YOLO loss layer. will be used during trainning """

    def __init__(self, args, cls_count=None):
        super(YoloDetectionLayer, self).__init__()

        self.args = args
        self.anchors = self.args.anchors
        self.num_anchors = self.args.num_anchors_per_resolution
        self.num_classes = self.args.num_classes
        self.bbox_attrib = 5 + self.args.num_classes
        self.img_size = self.args.img_size

        self.cls_count = cls_count

        # thresholds
        self.iou_threshold = 0.5
        self.lambda_coord = 5
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, backbone_out, gt_targets=None):
        """
        :param backbone_out: output of backbone. these are the features over which yolo calculates predictions
        :param gt_targets: ground truth bboxs. != None only during training.
        :return: yoloV3 loss and its components
        """

        # get backbone_out grid size and scale anchors
        batch_size = backbone_out.shape[0]
        in_w = backbone_out.shape[2]
        in_h = backbone_out.shape[3]

        scale_w = self.img_size[0] / in_w
        scale_h = self.img_size[1] / in_h
        #anchor size on output plane - downsampled the same as original image
        scaled_anchors = np.array([(a_w / scale_w, a_h / scale_h) for a_w, a_h in self.anchors])

        if scale_w == 32:
            anchors_idx = [0, 1, 2]
        elif scale_w == 16:
            anchors_idx = [3, 4, 5]
        elif scale_w == 8:
            anchors_idx = [6, 7, 8]

        # reshaping backbone predictions to (batch_size, #anchors, h, w, bbox_attributes)
        preds = backbone_out.view(batch_size, self.args.num_anchors_per_resolution, self.bbox_attrib, in_h, in_w)
        preds = preds.permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        t_x = torch.sigmoid(preds[..., 0])          # Center x
        t_y = torch.sigmoid(preds[..., 1])          # Center y
        t_w = preds[..., 2]                         # Width
        t_h = preds[..., 3]                         # Height
        pred_conf = torch.sigmoid(preds[..., 4])    # Objectness score
        pred_cls = torch.sigmoid(preds[..., 5:])    # Class predictions

        if gt_targets is not None:
            """ TRAINING """

            # build targets
            mask, noobj_mask, x0, y0, w, h, gt_conf, gt_cls = self.get_target(gt_targets, scaled_anchors, anchors_idx,
                                                                              in_w, in_h,
                                                                              self.iou_threshold)

            # calculating loss
            # bbox coordinate regression:
            loss_x = self.mse_loss(t_x[mask > 0], x0[mask > 0])
            loss_y = self.mse_loss(t_y[mask > 0], y0[mask > 0])
            loss_w = self.mse_loss(t_w[mask > 0], w[mask > 0])
            loss_h = self.mse_loss(t_h[mask > 0], h[mask > 0])

            n_obj = float((mask == 1).sum())
            n_no_obj = float((noobj_mask == 1).sum())

            loss_obj = self.bce_loss(pred_conf[mask > 0], gt_conf[mask > 0])
            loss_noobj = self.bce_loss(pred_conf[noobj_mask > 0],  gt_conf[noobj_mask > 0])

            mask_ = torch.cat(pred_cls.shape[-1] * [mask.unsqueeze(-1)], dim=-1)
            # cls weights
            if self.cls_count is not None:
                cls_weights = torch.tensor(1 / self.cls_count)
                cls_weights = cls_weights / cls_weights.sum()
            else:
                cls_weights = torch.ones(1, self.args.num_classes)

            weight_ = cls_weights.expand_as(mask_).cuda()

            loss_cls = nn.BCELoss(reduction='none')(pred_cls[mask_ > 0], gt_cls[mask_> 0])
            loss_cls = loss_cls * weight_[mask_>0]
            loss_cls = loss_cls.mean()

            loss_xy = loss_x + loss_y
            loss_wh = loss_w + loss_h

            bbox_loss = self.lambda_coord * (loss_xy + loss_wh)
            objectness_loss = self.lambda_obj * loss_obj + self.lambda_noobj * loss_noobj
            cls_loss = self.lambda_cls * loss_cls

            loss_total = bbox_loss + objectness_loss + cls_loss

            # this is for safety... just incase no gt targets are provided checking if loss is not nan due to no objects
            # if this happens, the only loss is no obj loss ( which should have indicated no objects in the image
            if (loss_total != loss_total):
                bbox_loss = torch.zeros_like(bbox_loss, requires_grad=True)
                objectness_loss = self.lambda_noobj * loss_noobj
                cls_loss = torch.zeros_like(cls_loss, requires_grad=True)
                loss_total = bbox_loss + objectness_loss + cls_loss

            return loss_total, bbox_loss, objectness_loss, cls_loss

        else:
            """ INFERENCE """

            # calculate offset of each cell on the grid
            c_column = torch.linspace(0, in_w-1, in_w).cuda()
            c_row = torch.linspace(0, in_h-1, in_h).cuda()

            # pytorch meshgrid default is row, column
            c_y, c_x = torch.meshgrid(c_row, c_column)      # shape of c_x and c_y (in_h, in_w)

            # expanding the shape of c_x, c_h to (batch_size, #anchors, in_h, in_w)
            c_x = c_x.repeat(batch_size, self.num_anchors, 1, 1)
            c_y = c_y.repeat(batch_size, self.num_anchors, 1, 1)

            # create tensors of anchor shapes
            anchors_w = torch.Tensor(scaled_anchors[anchors_idx[0]:anchors_idx[-1]+1, 0]).cuda()
            anchors_h = torch.Tensor(scaled_anchors[anchors_idx[0]:anchors_idx[-1]+1, 1]).cuda()

            # expnading the shape of anchors to (batch_size, #anchors, in_h, in_w)
            # each anchor shape is stored in different coordinate in #anchors dim: anchor_w[:,1,:,:]==anchor_w[:,2,:,:]
            a_w = anchors_w.repeat(batch_size, in_h, in_w, 1).permute(0, 3, 1, 2)
            a_h = anchors_h.repeat(batch_size, in_h, in_w, 1).permute(0, 3, 1, 2)

            # calculate the predicted bbox coordinates:
            # ---------------------------
            # b_x = sigmoid(t_x)+c_x     |
            # b_y = sigmoid(t_y)+c_y    |
            # b_w = a_w*exp(t_w)        |
            # b_h = a_h*exp(t_h)        |
            # ---------------------------

            pred_bboxes = preds[..., :4]
            pred_bboxes[..., 0] = t_x + c_x
            pred_bboxes[..., 1] = t_y + c_y
            pred_bboxes[..., 2] = a_w * torch.exp(t_w)
            pred_bboxes[..., 3] = a_h * torch.exp(t_h)

            # scaling the predicted boxes back to original image scale
            pred_bboxes[..., 0] *= scale_w
            pred_bboxes[..., 1] *= scale_h
            pred_bboxes[..., 2] *= scale_w
            pred_bboxes[..., 3] *= scale_h

            # fixing size of tensors for concatenation
            if len(pred_conf.shape) == 4:
                pred_conf = pred_conf.unsqueeze(-1)

            if len(pred_cls.shape) == 4:
                pred_cls = pred_cls.unsqueeze(-1)

            # reshaping bboxes, pred_conf and pred_cls. important later for concatenation:
            pred_bboxes = pred_bboxes.view(batch_size, -1, 4)
            pred_conf = pred_conf.view(batch_size, -1, 1)
            pred_cls = pred_cls.view(batch_size, -1, self.num_classes)


            pred_out = torch.cat((pred_bboxes, pred_conf, pred_cls), -1)

            return pred_out

    def get_target(self, gt_targets, anchors, anchors_idx, in_w, in_h, iou_threshold):
        """
        return the ground truth targets (bboxes) on grid of backbone last layer
        :param gt_targets = [batch_size, max_num_bbox, x, y, w, h, cls_idx]

        """
        batch_size = gt_targets.shape[0]
        gt_targets_device = gt_targets.device
        mask = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False, device=gt_targets_device)
        noobj_mask = torch.ones_like(mask)
        t_x = torch.zeros_like(mask)
        t_y = torch.zeros_like(mask)
        t_w = torch.zeros_like(mask)
        t_h = torch.zeros_like(mask)
        t_conf = torch.zeros_like(mask, device=gt_targets_device)
        t_cls = torch.zeros(batch_size, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False, device=gt_targets_device)

        anchors_idx = np.array(anchors_idx)
        # filling above tensors with gt_targets data
        for b in range(batch_size):
            for t in range(gt_targets.shape[1]):
                # verify targets exist
                if gt_targets[b, t].sum() <= 0:
                    continue
                gt_box = torch.zeros(1, 4)
                # convert position relative to grid cell
                g_x = gt_targets[b, t, 0] * in_w
                g_y = gt_targets[b, t, 1] * in_h
                g_w = gt_targets[b, t, 2] * in_w
                g_h = gt_targets[b, t, 3] * in_h

                # get grid cell indinces
                g_i = int(g_y)
                g_j = int(g_x)

                # create gt bbox
                gt_box[:, 2] = g_w
                gt_box[:, 3] = g_h

                # get shape of anchor box (w,h only)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((anchors.shape[0], 2)),
                                                                  anchors), 1))

                # calculate iou between gt and anchor shape
                anchors_ious = bbox_iou(gt_box, anchor_shapes)

                # set noobj mask to zeros where iou > threshold (ignore):
                noobj_mask[b, anchors_ious[anchors_idx] > iou_threshold, g_i, g_j] = 0

                # Find the best anchor
                best_n = np.argmax(anchors_ious)

                # updating masks to 1 only if anchor is in anchors_idx - which are anchors of current output resolution
                # this ensures that for every object, only a single anchor is found in all resolutions
                if best_n.item() in anchors_idx:
                    # selecting the indx of anchor in current resolution.
                    indx = list(range(len(anchors_idx)))
                    best_n = indx[np.where(anchors_idx == best_n.item())[0][0]]

                    # Masks
                    mask[b, best_n, g_i, g_j] = 1

                    # Setting the anchor gt bbox coordinates:
                    t_x[b, best_n, g_i, g_j] = g_x - g_j
                    t_y[b, best_n, g_i, g_j] = g_y - g_i

                    t_w[b, best_n, g_i, g_j] = torch.log(g_w / anchors[anchors_idx[best_n], 0] + 1e-16)
                    t_h[b, best_n, g_i, g_j] = torch.log(g_h / anchors[anchors_idx[best_n], 1] + 1e-16)

                    # Object
                    t_conf[b, best_n, g_i, g_j] = 1

                    # Class one hot encoding
                    t_cls[b, best_n, g_i, g_j, int(gt_targets[b, t, -1])] = 1

        return mask, noobj_mask, t_x, t_y, t_w, t_h, t_conf, t_cls

if __name__ == "__main__":
    """ test YOLOLoss """
    import argparse
    from YoloV3.nets.yolov3_tiny import YoloV3_tiny
    from YoloV3.dataloaders.kzir_dataloader import KzirDataset
    from torch.utils.data import Dataset, DataLoader
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2,
                        help="number of classes")
    parser.add_argument('--anchors', type=list, default=[[10, 13], [16, 30], [33, 23],
                                                         [30, 61], [62, 45], [59, 119],
                                                         [116, 90], [156, 198], [373, 326]],
                        help='anchors')
    parser.add_argument('--img-size', type=int, default=416,
                        nargs='+',
                        help='input img size')
    parser.add_argument('--database-root', type=str, default='/home/amit/Data/Kzir/Windows',
                        help='path to parent database root. its childern is images/ and /labels')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')

    args = parser.parse_args()

    if len([args.img_size]) == 1:
        args.img_size = [args.img_size, args.img_size]
    elif len(args.img_size) > 2:
        raise('wrong number of img size dimensions')


    x = torch.randn(args.batch_size, 3, args.img_size[0], args.img_size[1])

    backbone = YoloV3_tiny(args)
    backbone_out1, backbone_out2 = backbone(x)

    ''' testing inference '''
    ''' ----------------- '''
    yolo_layer = YoloDetectionLayer(args)
    loss = yolo_layer(backbone_out1)
    print("\nINFERENCE TEST PASSED!\n")

    ''' test training '''
    ''' ------------- '''
    # creating train dataset
    train_set = KzirDataset(args, split='')

    # creating train dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # looping over dataloader for sanity check
    pbar = tqdm.tqdm(train_loader)
    pbar.set_description("Training Progress")
    for i, sample in enumerate(pbar):
        image, bbox = sample["image"], sample["bbox"]
        # print("loaded image batch: {}; loaded bbox shape: {}".format(image.shape, bbox.shape))
        loss_total1, loss_xy1, loss_wh1, loss_cls1, loss_obj1, loss_noobj1 = yolo_layer(backbone_out1, bbox)
        loss_total2, loss_xy2, loss_wh2, loss_cls2, loss_obj2, loss_noobj2 = yolo_layer(backbone_out2, bbox)

    print("\nTRAINING TEST PASSED!\n")



