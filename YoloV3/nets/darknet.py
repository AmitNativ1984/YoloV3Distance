import torch
import torch.nn as nn
from collections import OrderedDict
import future
from tensorboardX import SummaryWriter
import logging
import hiddenlayer as hl
FORMAT = "[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger(__name__)

class BasicResBlock(nn.Module):
    """
    this part defines the basic resnet block. it assumes size of input and output has already been taken care of.
    there is no downsampling inside the block.
    """
    def __init__(self, inplanes, planes, skip_connection=True):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes[0],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels=planes[0], out_channels=planes[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

        self.skip_connection = skip_connection

    def forward(self, x):
        residual = x

        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.batch_norm2(y)

        y = y + residual

        out = self.relu2(y)
        return out

class DarkNet(nn.Module):
    def __init__(self, res_layers, num_classes):
        super(DarkNet, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = 3

        self.output_depth = self.num_anchors * (4 + 1 + self.num_classes)

        self.res_block1 = self.res_layer(inplanes=64, planes=[32, 64], num_blocks=res_layers[0]).cuda()
        self.res_block2 = self.res_layer(inplanes=128, planes=[64, 128], num_blocks=res_layers[1]).cuda()
        self.res_block3 = self.res_layer(inplanes=256, planes=[128, 256], num_blocks=res_layers[2]).cuda()
        self.res_block4 = self.res_layer(inplanes=512, planes=[256, 512], num_blocks=res_layers[3]).cuda()
        self.res_block5 = self.res_layer(inplanes=1024, planes=[512, 1024], num_blocks=res_layers[4]).cuda()

    def conv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        a repeating conv2d-batchNorm-leakyRelu block in yolo graph
        """
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding)
        batch_norm = nn.BatchNorm2d(out_channels)
        relu = nn.LeakyReLU(0.1)

        conv_layer = nn.Sequential(conv1,
                                   batch_norm,
                                   relu)

        return conv_layer.cuda()

    def res_layer(self, inplanes, planes, num_blocks):
        layers = []
        for res_block in range(num_blocks):
            layers.append(("Res_{}".format(res_block), BasicResBlock(inplanes, planes)))

        return nn.Sequential(OrderedDict(layers)).cuda()

    def conv2d_block(self, in_channels, channels, num_blocks):
        layers = []
        for conv_block in range(num_blocks):
            if conv_block == 0:
                conv_layer1 = self.conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=1, stride=1, padding=0)
            else:
                conv_layer1 = self.conv2d(in_channels=channels[1], out_channels=channels[0], kernel_size=1, stride=1, padding=0)

            layers.append(("conv2d_block_{}_{}".format(conv_block, 1), conv_layer1))
            conv_layer2 = self.conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=1, padding=1)
            layers.append(("conv2d_block_{}_{}".format(conv_block, 2), conv_layer2))

        return nn.Sequential(OrderedDict(layers)).cuda()

    def forward(self, x):
        """ darknet body """
        x = self.conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)(x)
        x = self.conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)(x)
        x = self.res_block1(x)

        x = self.conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)(x)
        x = self.res_block2(x)

        x = self.conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)(x)
        HighResFeatures = self.res_block3(x)

        x = self.conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)(HighResFeatures)
        MedResFeatures = self.res_block4(x)

        x = self.conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)(MedResFeatures)
        LowResFeatures = self.res_block5(x)

        """ multiscale fusion and detection """

        # Low scale output
        LowResFeatures = self.conv2d_block(in_channels=1024, channels=[512, 1024], num_blocks=3)(LowResFeatures)
        out_LowScale = self.conv2d(in_channels=1024, out_channels=self.output_depth, kernel_size=1, stride=1, padding=0)(LowResFeatures)

        # medium scale output
        x = self.conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)(LowResFeatures)
        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat((MedResFeatures, x), dim=1)
        x = self.conv2d_block(in_channels=768, channels=[256, 512], num_blocks=3)(x)
        out_MedScale = self.conv2d(in_channels=512, out_channels=self.output_depth, kernel_size=1, stride=1, padding=0)(x)

        # high scale output
        x = self.conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = torch.cat((HighResFeatures, x), dim=1)
        x = self.conv2d_block(in_channels=384, channels=[128, 256], num_blocks=3)(x)
        out_HighScale = self.conv2d(in_channels=256, out_channels=self.output_depth, kernel_size=1, stride=1, padding=0)(x)

        return out_LowScale, out_MedScale, out_HighScale


def darknet53(path_to_pretrained_weights=None, num_classes=3):
    """
    construct darknet 53 model (orignal net for YoLoV3)
    """

    model = DarkNet(res_layers=[1, 2, 8, 8, 4], num_classes=num_classes)
    logging.info("Successfully created DarkNet53 model")

    if path_to_pretrained_weights is not None:
        model.load_state_dict(torch.load(path_to_pretrained_weights))
        logging.info("Successfully loaded pre-trained weights from: " + path_to_pretrained_weights)

    return model

if __name__ == "__main__":
    writer = SummaryWriter()
    torch.cuda.set_device(0)

    model = darknet53().cuda()
    # print(model)

    dummy_input = torch.rand([1, 3, 416, 416]).cuda()

    low_scale, med_scale, high_scale = model(dummy_input)

    logging.info('finished dummy run')
    writer.close()
