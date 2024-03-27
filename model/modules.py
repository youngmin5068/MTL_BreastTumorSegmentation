
"""
Defines modules for breast tumor classification and segmentation model.
Use Interpretable classifier https://arxiv.org/abs/2002.07613
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.CBAM import Topt_CBAM
import model.tools as tools
from torchvision.models.resnet import conv3x3

class DoubleConv(nn.Module):


    def __init__(self, in_channels, out_channels, mid_channels=None,down=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if down:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,stride=2, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),

            )

    def forward(self, x):
        return (self.double_conv(x))
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels,down=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    

class BasicBlockV2(nn.Module):
    """
    Basic Residual Block of ResNet V2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)

        self.bn1 = nn.InstanceNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class BasicBlockV1(nn.Module):
    """
    Basic Residual Block of ResNet V1
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResNetV2(nn.Module):

    def __init__(self,input_channels,n_classes):
        super(ResNetV2, self).__init__()
        
        self.inc = DoubleConv(input_channels, 32,down=False)
        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.cbam0 = Topt_CBAM(32,0.005)
        self.cbam1 = Topt_CBAM(64,0.005)
        self.cbam2 = Topt_CBAM(128,0.005)
        self.cbam3 = Topt_CBAM(256,0.005)

        self.up1 = (Up(512, 256))
        self.up2 = (Up(256, 128))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(64, 32))

        self.outc = (OutConv(32, n_classes))

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, self.cbam3(x4))
        x = self.up2(x, self.cbam2(x3))
        x = self.up3(x, self.cbam1(x2))
        x = self.up4(x, self.cbam0(x1))


        x = self.outc(x)

        return x5,x

class AbstractMILUnit:

    def __init__(self, parameters, parent_module):
        self.parameters = parameters
        self.parent_module = parent_module


class TopTPercentAggregationFunction(AbstractMILUnit):
    """
    An aggregator that uses the SM to compute the y_global.
    Use the sum of topK value
    """
    def __init__(self, parameters, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(parameters, parent_module)
        self.percent_t = parameters["percent_t"]
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)



class PostProcessingStandard(nn.Module):

    def __init__(self, parameters):
        super(PostProcessingStandard, self).__init__()
        # map all filters to output classes
        self.gn_conv_last = nn.Conv2d(parameters["post_processing_dim"],
                                      parameters["num_classes"],
                                      (1, 1), bias=False)

    def forward(self, x_out):
        out = self.gn_conv_last(x_out)
        return torch.sigmoid(out)


class GlobalNetwork(AbstractMILUnit):

    def __init__(self, parameters, parent_module):
        super(GlobalNetwork, self).__init__(parameters, parent_module)
        # downsampling-branch

        self.downsampling_branch = ResNetV2(input_channels=1,n_classes=1)
        # post-processing
        self.postprocess_module = PostProcessingStandard(parameters)

    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.postprocess_module

    def forward(self, x):

        last_feature_map,output = self.downsampling_branch.forward(x)

        cam = self.postprocess_module.forward(last_feature_map)
        return last_feature_map, cam , output


class RetrieveROIModule(AbstractMILUnit):

    def __init__(self, parameters, parent_module):
        super(RetrieveROIModule, self).__init__(parameters, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        self.gpu_number = None if parameters["device_type"]!="gpu" else parameters["gpu_number"]

    def forward(self, x_original, cam_size, h_small):

        # retrieve parameters
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()

        # make sure that the size of h_small == size of cam_size
        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # combine channels
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = tools.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = tools.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()
    

class ResNetV1(nn.Module):

    def __init__(self, initial_filters, block, layers, input_channels=1):

        self.inplanes = initial_filters
        self.num_layers = len(layers)
        super(ResNetV1, self).__init__()

        # initial sequence
        # the first sequence only has 1 input channel which is different from original ResNet
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(initial_filters)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual sequence
        for i in range(self.num_layers):
            num_filters = initial_filters * pow(2,i)
            num_stride = (1 if i == 0 else 2)
            setattr(self, 'layer{0}'.format(i+1), self._make_layer(block, num_filters, layers[i], stride=num_stride))
        self.num_filter_last_seq = initial_filters * pow(2, self.num_layers-1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # first sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual sequences
        for i in range(self.num_layers):
            x = getattr(self, 'layer{0}'.format(i+1))(x)
        return x


class LocalNetwork(AbstractMILUnit):

    def add_layers(self):

        self.parent_module.dn_resnet = ResNetV1(64, BasicBlockV1, [2,2,2,2], 3)

    def forward(self, x_crop):

        # forward propagte using ResNet
        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1 , -1))
        # global average pooling
        res = res.mean(dim=2).mean(dim=2)
        return res
    

class AttentionModule(AbstractMILUnit):

    def add_layers(self):

        # The gated attention mechanism
        self.parent_module.mil_attn_V = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_U = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)
        # classifier
        self.parent_module.classifier_linear = nn.Linear(512, self.parameters["num_classes"], bias=False)

    def forward(self, h_crops):

        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(self.parent_module.mil_attn_U(h_crops_reshape)) * \
                          torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))
        attn_score = self.parent_module.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        y_crops = torch.sigmoid(self.parent_module.classifier_linear(z_weighted_avg))
        return z_weighted_avg, attn, y_crops

