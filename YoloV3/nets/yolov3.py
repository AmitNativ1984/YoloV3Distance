import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from YoloV3.nets.yolo_basic_blocks import conv2D_BN_LeakyRelu, MaxPoolPaddedStride1, ResBlock, ConvSet
from DeepTools.Onnx2TRT.onnx_ternsorRT import pytorch2onnx
from collections import OrderedDict

class darknet53(nn.Module):
    def __init__(self, args):
        super(darknet53, self).__init__()

        self.conv1 = conv2D_BN_LeakyRelu(inplanes=3, outplanes=32, kernelsize=3, stride=1, padding=1)
        self.resBlock = ResBlock

        self.layer1 = self._make_layer(convInPlanes=32, convOutPlanes=64, ResBlockPlanes=[32, 64], num_blocks=1)
        self.layer2 = self._make_layer(convInPlanes=64, convOutPlanes=128, ResBlockPlanes=[64, 128], num_blocks=2)
        self.layer3 = self._make_layer(convInPlanes=128, convOutPlanes=256, ResBlockPlanes=[128, 256], num_blocks=8)
        self.layer4 = self._make_layer(convInPlanes=256, convOutPlanes=512, ResBlockPlanes=[256, 512], num_blocks=8)
        self.layer5 = self._make_layer(convInPlanes=512, convOutPlanes=1024, ResBlockPlanes=[512, 1024], num_blocks=4)

    def _make_layer(self, convInPlanes, convOutPlanes, ResBlockPlanes, num_blocks):   #inplanes = 32, resBlock = [32, 64]
        layers = []

        # downsample convolution:
        layers.append(("ds_conv",
                       conv2D_BN_LeakyRelu(inplanes=convInPlanes, outplanes=convOutPlanes, kernelsize=3, stride=2, padding=1)))

        # resblocks:
        for block in range(0, num_blocks):
            layers.append(("res_block{}".format(block),
                           self.resBlock(inplanes=convOutPlanes, planes=ResBlockPlanes)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

class YoloV3(nn.Module):
    def __init__(self, args):
        super(YoloV3, self).__init__()

        self.num_classes = args.num_classes
        self.num_anchors = args.num_anchors_per_resolution
        self.num_output_features = self.num_anchors * (4 + 1 + self.num_classes)

        self.darknet53 = darknet53(args)

        self.conv5 = conv2D_BN_LeakyRelu(inplanes=512, outplanes=256, kernelsize=1, stride=1, padding=0)
        self.conv4 = conv2D_BN_LeakyRelu(inplanes=256, outplanes=128, kernelsize=1, stride=1, padding=0)

        self.conv_Set3 = ConvSet(inplanes=384, planes=[128, 256])
        self.conv_Set4 = ConvSet(inplanes=768, planes=[256, 512])
        self.conv_Set5 = ConvSet(inplanes=1024, planes=[512, 1024])

        self.conv_Out5 = nn.Sequential(conv2D_BN_LeakyRelu(inplanes=512, outplanes=1024, kernelsize=3, stride=1, padding=1),
                                      conv2D_BN_LeakyRelu(inplanes=1024, outplanes=self.num_output_features, kernelsize=1, stride=1, padding=0))

        self.conv_Out4 = nn.Sequential(conv2D_BN_LeakyRelu(inplanes=256, outplanes=512, kernelsize=3, stride=1, padding=1),
                                        conv2D_BN_LeakyRelu(inplanes=512, outplanes=self.num_output_features, kernelsize=1, stride=1, padding=0))

        self.conv_Out3 = nn.Sequential(conv2D_BN_LeakyRelu(inplanes=128, outplanes=256, kernelsize=3, stride=1, padding=1),
                                        conv2D_BN_LeakyRelu(inplanes=256, outplanes=self.num_output_features, kernelsize=1, stride=1, padding=0))
    def forward(self, x):
        y3, y4, y5 = self.darknet53(x)

        # output of low resolution...:
        y5 = self.conv_Set5(y5)
        out5 = self.conv_Out5(y5)

        # concatenating y4 - y5:
        y5 = self.conv5(y5)
        y5 = nn.Upsample(scale_factor=2, mode="nearest")(y5)

        # output of med resolution...
        y4 = torch.cat(tensors=(y4, y5), dim=1)
        y4 = self.conv_Set4(y4)
        out4 = self.conv_Out4(y4)

        # concatenating..
        y4 = self.conv4(y4)
        y4 = nn.Upsample(scale_factor=2, mode="nearest")(y4)

        # output of high resolution...
        y3 = torch.cat(tensors=(y3, y4), dim=1)
        y3 = self.conv_Set3(y3)
        out3 = self.conv_Out3(y3)

        return out3, out4, out5


if __name__ == "__main__":
    """ testing network """
    from PIL import Image
    import argparse
    from torchviz import make_dot

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2,
                        help="number of classes")

    parser.add_argument('--num-anchors-per-resolution', type=int, default=3,
                        help="number of classes")

    args = parser.parse_args()

    x = torch.zeros(1, 3, 416, 416)
    model = YoloV3(args)


    x = torch.zeros(1, 3, 416, 416)
    x = x.float()

    out3, out4, out5 = model(x)
    graph = make_dot(model(x))
    graph.render('./output/net/yolov3')

    # model_onnx = torch.onnx.export(model, x, './output/net/test.onnx', verbose=True, opset_version=11)
    # pytorch2onnx(model, (1, 3, 416, 416), './output/net/yolov3.onnx')