import torch.nn as nn
from conf.global_settings import CHANNEL_SIZE


def convDown(_in, _out, kernel_size=3,
             padding=1, stride=1, inplace=True,
             maxPool=False, kernel_size_mp=2, stride_mp=2):
    ''' Function Definition for our convolutional layers '''
    layers = [nn.Conv2d(_in, _out, kernel_size=kernel_size,
                        padding=padding, stride=stride)]
    layers.append(nn.ReLU(inplace=inplace))
    if maxPool:
        layers.append(nn.MaxPool2d(kernel_size=kernel_size_mp,
                                   stride=stride_mp))
    return layers


def linearBlock(_in, _out, last=False,
                inplace=True, p=0.5, inplace_dr=False):
    ''' Blocks for the classifier '''
    layers = [nn.Linear(_in, _out)]
    if not last:
        layers.append(nn.ReLU(inplace=inplace))
        layers.append(nn.Dropout(p=p, inplace=inplace_dr))
    return layers


class VGG(nn.Module):
    def __init__(self,
                 inplace=True):
        super(VGG, self).__init__()

        # Making Features
        convSizes = [3, 64, 128, 256, 256, 512, 512, 512, 512]
        poolingBlock = [True, True, False, True, False, True, False, True]

        features = []
        for i in range(len(convSizes) - 1):
            features += convDown(convSizes[i], convSizes[i + 1],
                                 maxPool=poolingBlock[i])
        self.features = nn.Sequential(*features)

        # Making avgpool
        adaptivePoolSize = (7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=adaptivePoolSize)

        # Making classifier
        classificationSize = 10
        linearSizes = [(adaptivePoolSize[0] * adaptivePoolSize[1]) * convSizes[-1], 4096, 4096,
                       classificationSize]
        classifier = []
        for i in range(len(linearSizes) - 1):
            classifier += linearBlock(linearSizes[i], linearSizes[i + 1],
                                      (i + 1 == len(linearSizes) - 1))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        B, C, _, _ = x.shape
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B, -1)
        x = self.classifier(x)

        return