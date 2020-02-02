import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from YoloV3.nets.yolo_basic_blocks import conv2D_BN_LeakyRelu, MaxPoolPaddedStride1
from DeepTools.Onnx2TRT.onnx_ternsorRT import pytorch2onnx

class YoloV3_tiny(nn.Module):
    def __init__(self, args):
        super(YoloV3_tiny, self).__init__()

        self.num_classes = args.num_classes
        self.num_anchors = args.num_anchors_per_resolution

        self.num_output_features = self.num_anchors * (4 + 1 + self.num_classes)

        self.conv2D_BN_LeakyRelu1 = conv2D_BN_LeakyRelu(inplanes=3, outplanes=16, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu2 = conv2D_BN_LeakyRelu(inplanes=16, outplanes=32, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu3 = conv2D_BN_LeakyRelu(inplanes=32, outplanes=64, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu4 = conv2D_BN_LeakyRelu(inplanes=64, outplanes=128, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu5 = conv2D_BN_LeakyRelu(inplanes=128, outplanes=256, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu6 = conv2D_BN_LeakyRelu(inplanes=256, outplanes=512, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu7 = conv2D_BN_LeakyRelu(inplanes=512, outplanes=1024, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu8 = conv2D_BN_LeakyRelu(inplanes=1024, outplanes=256, kernelsize=1, stride=1, padding=0)
        self.conv2D_BN_LeakyRelu9 = conv2D_BN_LeakyRelu(inplanes=256, outplanes=512, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu10 = conv2D_BN_LeakyRelu(inplanes=512, outplanes=self.num_output_features, kernelsize=1, stride=1, padding=0)
        self.conv2D_BN_LeakyRelu11 = conv2D_BN_LeakyRelu(inplanes=256, outplanes=128, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu12 = conv2D_BN_LeakyRelu(inplanes=384, outplanes=256, kernelsize=3, stride=1, padding=1)
        self.conv2D_BN_LeakyRelu13 = conv2D_BN_LeakyRelu(inplanes=256, outplanes=self.num_output_features, kernelsize=1, stride=1, padding=0)

        self.MaxPoolPaddedStride1 = MaxPoolPaddedStride1()

    def forward(self, x):

        # input size = 416x416x3
        x = self.conv2D_BN_LeakyRelu1(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 208x208x16
        x = self.conv2D_BN_LeakyRelu2(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 104x104x32
        x = self.conv2D_BN_LeakyRelu3(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)


        # 52x52x64
        x = self.conv2D_BN_LeakyRelu4(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        # 26x26x128
        y0 = self.conv2D_BN_LeakyRelu5(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(y0)

        # 13x13x256
        x = self.conv2D_BN_LeakyRelu6(x)
        x = self.MaxPoolPaddedStride1(x)

        # 13x13x512
        x = self.conv2D_BN_LeakyRelu7(x)
        # 13x13x1024
        y1 = self.conv2D_BN_LeakyRelu8(x)
        # 13x13x256
        x = self.conv2D_BN_LeakyRelu9(y1)
        # 13x13x512
        out1 = self.conv2D_BN_LeakyRelu10(x)
        # 13x13xnum_ouput_features

        # 13x13x256
        y1 = self.conv2D_BN_LeakyRelu11(y1)
        y1 = nn.Upsample(scale_factor=2, mode="nearest")(y1)
        # 26x26x128
        y = torch.cat(tensors=(y0, y1), dim=1)
        # 26x26x384
        y = self.conv2D_BN_LeakyRelu12(y)
        # 26x26x256
        out2 = self.conv2D_BN_LeakyRelu13(y)
        # 26x26xnum_ouput_features

        return out1, out2


if __name__ == "__main__":
    """ testing network """
    from PIL import Image
    import argparse
    from torchviz import make_dot

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2,
                        help="number of classes")

    args = parser.parse_args()

    x = torch.zeros(1, 3, 416, 416)
    model = YoloV3_tiny(args)


    x = torch.zeros(1, 3, 416, 416)
    x = x.float()


    graph = make_dot(model(x))
    graph.render('./output/net/yolov3-tiny')

    # model_onnx = torch.onnx.export(model, x, './output/net/test.onnx', verbose=True, opset_version=11)
    pytorch2onnx(model, (1, 3, 416, 416), './output/net/yolov3_tiny.onnx')