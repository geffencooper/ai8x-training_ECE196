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
import torch
import distiller.apputils as apputils

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

class GeffenNetP2(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=2, num_channels=1, dimensions=(64, 64),
                 planes=60, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()
        self.feature_extractor = GeffenNet(**kwargs)
        
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(self.feature_extractor, "../ai8x-synthesis/trained/geffen_classifier_q.pth.tar")
        #self.feature_extractor.load_state_dict(torch.load("../ai8x-synthesis/trained/geffen_classifier.pth.tar"))
        self.feature_extractor = model
       
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.fc1 = ai8x.Linear(10*4*4, 8, bias=True, wide=True, **kwargs)
        
        #print(self.feature_extractor)
        #n_sizes = self._get_conv_output(input_shape)

        #self.classifier = nn.Linear(n_sizes, num_classes)
       
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 256
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
       
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self._forward_features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)

        return x
    
def geffennet_p2(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return GeffenNetP2(**kwargs)

def geffennet_p3(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return GeffenNetP3(**kwargs)

class GeffenNetP3(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=2, num_channels=1, dimensions=(64, 64),
                 planes=60, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()
        self.feature_extractor = GeffenNet(**kwargs)
        
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(self.feature_extractor, "../ai8x-synthesis/trained/classifier_q.pth.tar")
        #self.feature_extractor.load_state_dict(torch.load("../ai8x-synthesis/trained/geffen_classifier.pth.tar"))
        self.feature_extractor = model
       
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.fc1 = ai8x.Linear(10*4*4, 4, bias=True, wide=True, **kwargs)
        
        #print(self.feature_extractor)
        #n_sizes = self._get_conv_output(input_shape)

        #self.classifier = nn.Linear(n_sizes, num_classes)
       
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    
    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 256
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
       
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self._forward_features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)

        return x
    
class GeffNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, dimensions=(128, 128),
                 planes=10, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # 1x128x128 --> 10x128x128
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 10, 3, padding=1, bias=False, **kwargs)
        
        # 10x128x128 --> 20x64x64
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(10,20, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        
        # 20x64x64 --> 30x32x32
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(20,30, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        
        # 30x32x32 --> 30x16x16
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        
        # 30x16x16 --> 30x8x8
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        
        # 30x8x8 --> 30x10x10
        self.conv6 = ai8x.FusedConv2dReLU(30, 30, 3, padding=2, bias=False,**kwargs)
        
        # 30x10x10 --> 30x5x5
        self.conv7 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
       
        self.fc1 = ai8x.Linear(30*5*5, 2, bias=True, wide=True, **kwargs)

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
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
    
def geffnet(pretrained=False, **kwargs):
    assert not pretrained
    return GeffNet(**kwargs)


models = [
    {
        'name': 'geffennet',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'geffennet_p2',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'geffennet_p3',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'geffnet',
        'min_input': 1,
        'dim': 2,
    }
]
