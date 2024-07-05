# Code extracted from https://github.com/doiken23/DeepLab_pytorch/blob/master/DeepLab_v2_vgg.py

import torch.nn as nn

from deeplab_pytorch.libs.models.resnet import _ResLayer, _Stem

def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                    stride=1, padding=rate, dilation=rate),
                                 # nn.BatchNorm2d(planes, eps=1e-5, momentum=1 - 0.999),
                                 nn.ReLU())
    return conv3x3_relu

class VGG16_feature(nn.Module):
    def __init__(self):
        super(VGG16_feature, self).__init__()

        self.features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1))
        self.features2 = nn.Sequential(conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       nn.MaxPool2d(3, stride=1, padding=1))

    def forward(self, x):
        x = self.features(x)
        x = self.features2(x)

        return x

"""
class VGG16_feature(nn.Module):
    def __init__(self):
        super(VGG16_feature, self).__init__()

        self.features = nn.Sequential(_Stem(64),
                                      _ResLayer(3, 64, 256, 1, 1),
                                      _ResLayer(4, 256, 512, 2, 1),
                                      _ResLayer(23, 512, 1024, 1, 2),
                                      _ResLayer(3, 1024, 2048, 1, 4))

    def forward(self, x):
        x = self.features(x)
        return x"""