import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict
import re
import numpy as np
import torchvision.models as models

import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
           
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        #x = x1 + x2
        x = self.conv_bn_relu(x)
        return x  

'''
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Wx = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size = 1),nn.BatchNorm2d(out_channels))
        self.conv_bn_relu= nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels))
        self.conv_sigmoid= nn.Sequential(nn.Conv2d(out_channels, 1, kernel_size=1, padding=0), nn.BatchNorm2d(1),nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.Wx(x2)
        sum = x1 + x2
        x = self.relu(sum)
        att = self.conv_sigmoid(x)
        mul = att*sum
        x = self.conv_bn_relu(mul)
        return x  
'''
###################################   DTC   ##############################################
class DTCResNet(nn.Module):  

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(DTCResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        channels = [64,64,128,256,512]
        self.out2 = nn.Conv2d(64, num_classes, 1)
        self.tanh = nn.Tanh()
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, num_classes, kernel_size=1,bias=False))
        self.dist = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, num_classes, kernel_size=1,bias=False))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        encoder = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        
        x = self.layer2(x)
        encoder.append(x)
        
        x = self.layer3(x)
        encoder.append(x)
        
        x = self.layer4(x)
        encoder.append(x)
        
        d4 = self.decode4(encoder[4], encoder[3])
        d3 = self.decode3(d4, encoder[2]) 
        d2 = self.decode2(d3, encoder[1]) 
        d1 = self.decode1(d2, encoder[0]) 
        out = self.decode0(d1)     
        out2 = self.dist(d1)
        tanh_out = self.tanh(out2)
        return tanh_out,out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def DTC(num_class) :
    return DTCResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_class)
    
    
################################   UNet   ######################################

class ResNet(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        channels = [64,64,128,256,512]
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1,bias=False))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        encoder = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        
        x = self.layer2(x)
        encoder.append(x)
        
        x = self.layer3(x)
        encoder.append(x)
        
        x = self.layer4(x)
        encoder.append(x)
        
        d4 = self.decode4(encoder[4], encoder[3]) 
        d3 = self.decode3(d4, encoder[2]) 
        d2 = self.decode2(d3, encoder[1]) 
        d1 = self.decode1(d2, encoder[0]) 
        out = self.decode0(d1)    
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet34(num_class) :
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes = num_class)
    

################################   Discriminator   ######################################
class Discriminator(nn.Module):

    def __init__(self, num_classes, map_channel, ndf=16, n_channel=3):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Conv2d(map_channel, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*6, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ndf*6, ndf*8, kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Linear(ndf*8, num_classes)
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, mask, image):
        mask_feature = self.conv0(mask)
        image_feature = self.conv1(image)
        
        x = torch.add(mask_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        x = self.conv5(x)
        x = self.leaky_relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



################################   UNet++   ######################################
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Unetpp(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(Unetpp, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        
        self.deep_supervision = False
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        nb_filter = [64,64,128,256,512]
        self.up1 = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.up6 = nn.ConvTranspose2d(nb_filter[0], nb_filter[0], kernel_size=2, stride=2)
        self.up7 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)
        self.up8 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)
        self.up9 = nn.ConvTranspose2d(nb_filter[2], nb_filter[0], kernel_size=2, stride=2)
        self.up10 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
                
        self.conv0_1 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x0_0 = self.relu(x)
        
        x = self.maxpool(x)
        x1_0 = self.layer1(x)
        
        x2_0 = self.layer2(x1_0)
    
        x3_0 = self.layer3(x2_0)
        
        x4_0 = self.layer4(x3_0)
        
        x0_1 = self.conv0_1(torch.add(x0_0, self.up1(x1_0)))

        x1_1 = self.conv1_1(x1_0+self.up2(x2_0))
        x0_2 = self.conv0_2(x0_0+x0_1+self.up3(x1_1))

        x2_1 = self.conv2_1(x2_0+ self.up4(x3_0))
        x1_2 = self.conv1_2(x1_0+ x1_1+self.up5(x2_1))
        x0_3 = self.conv0_3(x0_0+ x0_1+x0_2+self.up6(x1_2))

        x3_1 = self.conv3_1(x3_0+ self.up7(x4_0))
        x2_2 = self.conv2_2(x2_0+ x2_1+self.up8(x3_1))
        x1_3 = self.conv1_3(x1_0+ x1_1+x1_2+self.up9(x2_2))
        x0_4 = self.conv0_4(x0_0+ x0_1+x0_2+ x0_3+self.up10(x1_3))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output      

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def UNet_plus(num_class) :
    return Unetpp(BasicBlock, [3, 4, 6, 3],num_classes = num_class)



################################   DeepLab   ######################################
class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()
        ASPP_out = 128
        self.conv_1x1_1 = nn.Conv2d(256, ASPP_out, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(ASPP_out)
 
        self.conv_3x3_1 = nn.Conv2d(256, ASPP_out, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(ASPP_out)
 
        self.conv_3x3_2 = nn.Conv2d(256, ASPP_out, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(ASPP_out)
 
        self.conv_3x3_3 = nn.Conv2d(256, ASPP_out, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(ASPP_out)
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(256, ASPP_out, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(ASPP_out)
 
        self.conv_1x1_3 = nn.Conv2d(ASPP_out*5, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
 
        self.out_conv = nn.Conv2d(256, num_classes, kernel_size=1)
    
        self.concat_conv = nn.Conv2d(256+128, 256, kernel_size=3, padding=1)
        self.concat_bn = nn.BatchNorm2d(256)
        self.mid_conv1 = nn.Conv2d(64, 128, kernel_size=1)
        self.mid_bn = nn.BatchNorm2d(128)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    def forward(self, feature_map, mid): 
        feature_map_h = feature_map.size()[2] # (h/8)
        feature_map_w = feature_map.size()[3] # (w/8)
 
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) 
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) 
 
        out_img = self.avg_pool(feature_map) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 
        
        mid_feature = F.relu(self.mid_bn(self.mid_conv1(mid))) 
        #print(mid_feature.shape)
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 
        out_up = self.up(out)
        #print(out_up.shape)
        concat = torch.cat([out_up, mid_feature], 1) 
        concat_out = F.relu(self.concat_bn(self.concat_conv(concat)))
        concat_up = self.up(concat_out)
        #print(concat_up.shape)
        out = self.out_conv(concat_up) 
        return out
    
class ResNet_deeplabv3(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(ResNet_deeplabv3, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.aspp = ASPP(num_classes)
        
    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        midle = x
        x = self.layer2(x)
        x = self.layer3(x)        
        out = self.aspp(x,midle)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def DeepLabv3(num_class) :
    return ResNet_deeplabv3(BasicBlock, [3, 4, 6, 3],num_classes = num_class)
    



################################   PSPNet   ######################################
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
    
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)
    
class ResNet_PSPNet(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(ResNet_PSPNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        
        sizes=(1, 2, 3, 6)
        self.psp = PSPModule(256, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 32)
        self.up_4 = PSPUpsample(32, 32)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
                
        p = self.psp(x)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        p = self.up_4(p)
        out = self.final(p)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
        
def PSPNet(num_class) :
    return ResNet_PSPNet(BasicBlock, [3, 4, 6, 3],num_classes = num_class)


################################   FCN   ######################################
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        pretrained_net = models.resnet34(pretrained=False)
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) 
        self.stage2 = list(pretrained_net.children())[-4] #
        self.stage3 = list(pretrained_net.children())[-3] #
        
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4) 
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = self.bilinear_kernel(num_classes, num_classes, 4) 

        
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s
    
    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
       
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)



################################   URPC   ######################################
class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x
    
def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)
        
class UpBlock(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return [dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg]

class ResNet_URPC(nn.Module):

    def __init__(self,block, layers,num_classes,zero_init_residual=False,groups = 1,width_per_group = 64): 
        super(ResNet_URPC, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        params = {'in_chns': 3,
                  'feature_chns': [64,64,128,256,512],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.decoder = Decoder_URPC(params)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        shape = x.shape[2:]
        
        encoder = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        
        x = self.layer2(x)
        encoder.append(x)
        
        x = self.layer3(x)
        encoder.append(x)
        
        x = self.layer4(x)
        encoder.append(x)
        
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(encoder, shape)
        
        return [dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def URPC(num_class) :
    
    return ResNet_URPC(BasicBlock, [3, 4, 6, 3],num_classes = num_class)


################################   VGG16   ######################################
class VGG16(nn.Module):
    def __init__(self,num_classes, batch_norm,init_weights: bool = True) -> None:
        super(VGG16, self).__init__()
        BN = batch_norm
        self.layer0 = self._make_layers(3, [64, 64, 'M'], BN)
        self.layer1 = self._make_layers(64, [128, 128, 'M'], BN)
        self.layer2 = self._make_layers(128, [256, 256, 256, 'M'], BN)
        self.layer3 = self._make_layers(256, [512, 512,512, 'M'], BN)
        self.layer4 = self._make_layers(512, [512, 512,512, 'M'], BN)
        
        channels = [64,128,256,512,512]
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, num_classes, kernel_size=1,bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder = []
        x = self.layer0(x)
        encoder.append(x)
        
        x = self.layer1(x)
        encoder.append(x)
        
        x = self.layer2(x)
        encoder.append(x)
        
        x = self.layer3(x)
        encoder.append(x)
        
        x = self.layer4(x)
        encoder.append(x)
        
        d4 = self.decode4(encoder[4], encoder[3])
        d3 = self.decode3(d4, encoder[2]) 
        d2 = self.decode2(d3, encoder[1]) 
        d1 = self.decode1(d2, encoder[0]) 
        out = self.decode0(d1)     
        return out
            
    def _make_layers(self,in_channel, cfg, batch_norm):
        in_channels = in_channel
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)



################################   MobileNet   ######################################
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(self,in_planes: int,out_planes: int,kernel_size: int = 3,stride: int = 1,
                 groups: int = 1,norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None, dilation: int = 1) -> None:
        
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True))
        self.out_channels = out_planes
# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation
class InvertedResidual(nn.Module):
    def __init__(self,inp: int,oup: int,stride: int, expand_ratio: int,norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 2,width_mult: float = 1.0,inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,block: Optional[Callable[..., nn.Module]] = None,norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
       
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        channels = [16,24,32,96,1280]
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, num_classes, kernel_size=1,bias=False))

    def _forward_impl(self, x: Tensor) -> Tensor:
        oriencoder = []
        encoder = []
        for feature in self.features:
            x = feature(x)
            oriencoder.append(x) 
            
        encoder.append(oriencoder[1]) 
        encoder.append(oriencoder[3])  
        encoder.append(oriencoder[6])  
        encoder.append(oriencoder[13])  
        encoder.append(oriencoder[18])  
            
        d4 = self.decode4(encoder[4], encoder[3])
        d3 = self.decode3(d4, encoder[2])
        d2 = self.decode2(d3, encoder[1])
        d1 = self.decode1(d2, encoder[0]) 
        out = self.decode0(d1)     
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


################################   DenseNet   ######################################
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int,growth_rate: int,bn_size: int,
        drop_rate: float,memory_efficient: bool = False) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *growth_rate, kernel_size=1, stride=1, bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:  
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int,
        growth_rate: int,drop_rate: float,memory_efficient: bool = False) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self,growth_rate: int = 32, block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,bn_size: int = 4,drop_rate: float = 0,num_classes: int = 1000,memory_efficient: bool = False) -> None:
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        channels = [16,64,128,256,256]
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, num_classes, kernel_size=1,bias=False))
        
    def forward(self, x: Tensor) -> Tensor:
        oriencoder = []
        encoder = []
        for encode in self.features:
            x = encode(x)
            oriencoder.append(x)
            
        encoder.append(oriencoder[2]) 
        encoder.append(oriencoder[4])  
        encoder.append(oriencoder[6])  
        encoder.append(oriencoder[8])  
        encoder.append(oriencoder[11])
        
        d4 = self.decode4(encoder[4], encoder[3])
        d3 = self.decode3(d4, encoder[2])
        d2 = self.decode2(d3, encoder[1])
        d1 = self.decode1(d2, encoder[0]) 
        out = self.decode0(d1)     
        return out

def _densenet(arch: str,growth_rate: int,block_config: Tuple[int, int, int, int],num_init_features: int,
    pretrained: bool,progress: bool,**kwargs: Any) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model

def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:

    return _densenet('densenet121', 8, (6, 12, 24, 16), 16, pretrained, progress, **kwargs)
