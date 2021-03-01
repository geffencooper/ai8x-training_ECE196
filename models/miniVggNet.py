###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

import torch.nn as nn
import ai8x
import numpy as np
import torch
import distiller.apputils as apputils

# this architecture tries to follow the general idea of VGG but with smaller input and less layers
# basically does conv3x3 then pool but doubles the channels at each 'stage'
# 80x80 was chosen because divides down to 5x5 but doesn't require streaming
class MiniVggNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, dimensions=(80, 80),
                 planes=8, pool=2, fc_inputs=2, bias=False, **kwargs):
        super().__init__()

         # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # 1x80x80 --> 8x80x80 (padding by 1 so same dimension)
        self.conv1 = ai8x.FusedConv2dReLU(1, 8, 3, padding=1, bias=False, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(8, 8, 3, padding=1, bias=False, **kwargs)
        
        # 8x80x80 --> 16x40x40 (padding by 1 so same dimension)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8,16, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(16, 16, 3, padding=1, bias=False, **kwargs)
        
        # 16x40x40 --> 32x20x20 (padding by 1 so increase dimension)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(16,32, 3, padding=1, bias=False,pool_size=2, pool_stride=2, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(32, 32, 3, padding=1, bias=False, **kwargs)
        
        # 32x20x20 --> 64x12x12 (padding by 2 so increase dimension)
        self.conv7 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, padding=2, bias=False, pool_size=2, pool_stride=2,**kwargs)
        self.conv8 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=False, **kwargs)
        
        # 64x12x12 --> 64x6x6 (padding by 1 so same dimension)
        self.conv9 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        
        # 64x6x6 --> 64x3x3 (passing by 1 so same dimension)
        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, padding=1, bias=False, pool_size=2, pool_stride=2,**kwargs)
        
        # flatten to fully connected layer
        self.fc1 = ai8x.FusedLinearReLU(64*3*3, 10, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(10, 2, bias=True, wide=True, **kwargs)

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
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x= self.fc2(x)

        return x


def mini_vgg_net(pretrained=False, **kwargs):
    assert not pretrained
    return MiniVggNet(**kwargs)




# this network adds an additional output layer to mini_vgg_net to predict bounding boxes
class MiniVggNet_bb(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, dimensions=(80, 80),
                 planes=8, pool=2, fc_inputs=2, bias=False, **kwargs):
        super().__init__()
        
        # load the pretrained model
        self.feature_extractor = MiniVggNet(**kwargs)
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(self.feature_extractor, "../ai8x-synthesis/trained/mini_vgg_net.pth.tar")
        self.feature_extractor = model
       
        # freeze the weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # retrain the last layer to detect a bounding box
        self.feature_extractor.fc1 = ai8x.Linear(64*3*3, 4, bias=False, wide=True, **kwargs)
            
        # add a fully connected layer for bounding box detection after the conv10
        #self.fc3 = ai8x.Linear(64*3*3, 4, bias=True, wide=True, **kwargs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
        #self.feature_extractor.fc2 = ai8x.Linear(64*3*3, 4, bias=True, wide=True, **kwargs)
        
        #print(self.feature_extractor)
       
    # copy the forward prop of MiniVggNet but add layers
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.conv2(x)
        x = self.feature_extractor.conv3(x)
        x = self.feature_extractor.conv4(x)
        x = self.feature_extractor.conv5(x)
        x = self.feature_extractor.conv6(x)
        x = self.feature_extractor.conv7(x)
        x = self.feature_extractor.conv8(x)
        x = self.feature_extractor.conv9(x)
        x = self.feature_extractor.conv10(x)
        x = x.view(x.size(0), -1)
        
        # output layers
        x1 = self.feature_extractor.fc1(x) # only output a bb for now
        #x1 = self.feature_extractor.fc2(x1) # binary classifier
        
        #x2 = self.fc3(x) # bounding box

        #return x1,x2
        return x1
    
def mini_vgg_net_bb(pretrained=False, **kwargs):
    assert not pretrained
    return MiniVggNet_bb(**kwargs)


models = [
    {
        'name': 'mini_vgg_net',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'mini_vgg_net_bb',
        'min_input': 1,
        'dim': 2,
    }
]
