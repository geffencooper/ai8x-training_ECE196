###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Networks that fit into AI84

Optionally quantize/clamp activations
"""
import torch.nn as nn
import torch.nn.functional as F
import ai8x
import numpy as np
import matplotlib.pyplot as plt

class GeffenNet(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=2, num_channels=1, dimensions=(64, 64),
                 planes=60, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 30, 3, padding=1, bias=False, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(30,30, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(30,30, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(30, 10, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        
       
        self.fc1 = ai8x.Linear(10*4*4, 2, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def geffennet(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return GeffenNet(**kwargs)





models = [
    {
        'name': 'geffennet',
        'min_input': 1,
        'dim': 2,
    }
]
