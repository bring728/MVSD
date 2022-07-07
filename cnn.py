# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from utils import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self, cfg):
        super(ResUNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]

        # original
        layers = [3, 4, 4, 3, 2]
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        if cfg.input == 'rgb':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False, padding_mode='reflect')
        else:
            self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False, padding_mode='reflect')
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2, dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv5 = upconv(filters[4], filters[3], 3, 2)
        self.iconv5 = conv(filters[3] + filters[3], filters[3], 3, 1)
        self.upconv4 = upconv(filters[3], filters[2], 3, 2)
        self.iconv4 = conv(filters[2] + filters[2], filters[2], 3, 1)
        self.upconv3 = upconv(filters[2], filters[1], 3, 2)
        self.iconv3 = conv(filters[1] + filters[1], filters[1], 3, 1)
        self.upconv2 = upconv(filters[1], filters[0], 3, 2)
        self.iconv2 = conv(filters[0] + filters[0], cfg.dim, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(cfg.dim, cfg.dim, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    @autocast()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upconv4(x4)
        x = self.skipconnect(x3, x)
        x = self.iconv4(x)

        x = self.upconv3(x)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)
        return x_out


def make_layer(pad_type='zeros', padding=1, in_ch=3, out_ch=64, kernel=3, stride=1, num_group=4, act='relu', norm_layer='group'):
    act = act.lower()
    pad_type = pad_type.lower()
    norm_layer = norm_layer.lower()
    layers = []
    if pad_type == 'rep':
        padding_mode = 'replicate'
    elif pad_type == 'zeros':
        padding_mode = 'zeros'
    else:
        assert 'not implemented pad'
    layers.append(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, bias=norm_layer != 'batch',
                  padding_mode=padding_mode, padding=padding))

    if norm_layer == 'group':
        layers.append(nn.GroupNorm(num_groups=num_group, num_channels=out_ch))
    elif norm_layer == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm_layer == 'instance':
        layers.append(nn.InstanceNorm2d(out_ch))
    elif norm_layer == 'None':
        norm = 'none'
    else:
        assert 'not implemented pad'

    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'None':
        act = 'none'
    else:
        assert 'not implemented act'
    return nn.Sequential(*layers)


class NormalNet(nn.Module):
    def __init__(self, cfg):
        super(NormalNet, self).__init__()
        self.layer_d_1 = make_layer(pad_type='rep', in_ch=5, out_ch=64, kernel=4, stride=2, num_group=4, norm_layer=cfg.norm_layer)
        self.layer_d_2 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_6 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)

        self.layer_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_u_5 = make_layer(in_ch=256, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=cfg.norm_layer)
        self.layer_u_6 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=cfg.norm_layer)

        self.layer_final = make_layer(pad_type='rep', in_ch=64, out_ch=3, kernel=3, stride=1, act='None', norm_layer='None')

    @autocast()
    def forward(self, x):
        x1 = self.layer_d_1(x)
        x2 = self.layer_d_2(x1)
        x3 = self.layer_d_3(x2)
        x4 = self.layer_d_4(x3)
        x5 = self.layer_d_5(x4)
        x6 = self.layer_d_6(x5)

        dx1 = self.layer_u_1(x6)
        dx2 = self.layer_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = self.layer_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx4 = self.layer_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.layer_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx6 = self.layer_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        x_out = self.layer_final(dx6)

        x_out = torch.clamp(1.01 * torch.tanh(x_out), -1, 1)
        normal = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out, dim=1, keepdim=True)), min=1e-6)
        return normal


class DirectLightingNet(nn.Module):
    def __init__(self, cfg):
        super(DirectLightingNet, self).__init__()
        self.SGNum = cfg.SGNum

        self.layer_d_1 = make_layer(pad_type='rep', in_ch=8, out_ch=32, kernel=4, stride=2, num_group=2, norm_layer=cfg.norm_layer)
        self.layer_d_2 = make_layer(in_ch=32, out_ch=64, kernel=4, stride=2, num_group=4, norm_layer=cfg.norm_layer)
        self.layer_d_3 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_d_4 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_6 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_7 = make_layer(in_ch=512, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_8 = make_layer(in_ch=512, out_ch=1024, kernel=3, stride=1, num_group=64, norm_layer=cfg.norm_layer)

        activation_func = nn.ELU(inplace=True)
        self.layer_intensity = nn.Sequential(nn.Linear(1024, 128), activation_func,
                                             nn.Linear(128, 128), activation_func,
                                             nn.Linear(128, 128), activation_func,
                                             nn.Linear(128, cfg.SGNum * 3),
                                             )

        self.layer_axis_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_axis_u_2 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_axis_u_3 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_axis_u_4 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_axis_u_5 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_axis_final = make_layer(pad_type='rep', in_ch=128, out_ch=cfg.SGNum * 3, kernel=3, stride=1, act='None',
                                           norm_layer='None')

        self.layer_sharp_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_sharp_u_2 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_sharp_u_3 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_sharp_u_4 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_sharp_u_5 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_sharp_final = make_layer(pad_type='rep', in_ch=128, out_ch=cfg.SGNum, kernel=3, stride=1, act='None', norm_layer='None')

        self.layer_vis_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_vis_u_2 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_vis_u_3 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_vis_u_4 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_vis_u_5 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_vis_final = make_layer(pad_type='rep', in_ch=128, out_ch=cfg.SGNum, kernel=3, stride=1, act='None', norm_layer='None')

    @autocast()
    def forward(self, x):
        x1 = self.layer_d_1(x)
        x2 = self.layer_d_2(x1)
        x3 = self.layer_d_3(x2)
        x4 = self.layer_d_4(x3)
        x5 = self.layer_d_5(x4)
        x6 = self.layer_d_6(x5)
        x7 = self.layer_d_7(x6)
        x8 = self.layer_d_8(x7)
        x_color = F.adaptive_avg_pool2d(x8, (1, 1))[..., 0, 0]
        intensity = torch.tanh(self.layer_intensity(x_color))

        dx1 = self.layer_axis_u_1(x8)
        dx2 = self.layer_axis_u_2(F.interpolate(torch.cat([dx1, x7], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx2 = F.interpolate(dx2, [x6.size(2), x6.size(3)], mode='bilinear', align_corners=False)
        dx3 = self.layer_axis_u_3(F.interpolate(torch.cat([dx2, x6], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = F.interpolate(dx3, [x5.size(2), x5.size(3)], mode='bilinear', align_corners=False)
        dx4 = self.layer_axis_u_4(F.interpolate(torch.cat([dx3, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.layer_axis_u_5(dx4)
        axis_out = torch.tanh(self.layer_axis_final(dx5))

        dx1 = self.layer_sharp_u_1(x8)
        dx2 = self.layer_sharp_u_2(F.interpolate(torch.cat([dx1, x7], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx2 = F.interpolate(dx2, [x6.size(2), x6.size(3)], mode='bilinear', align_corners=False)
        dx3 = self.layer_sharp_u_3(F.interpolate(torch.cat([dx2, x6], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = F.interpolate(dx3, [x5.size(2), x5.size(3)], mode='bilinear', align_corners=False)
        dx4 = self.layer_sharp_u_4(F.interpolate(torch.cat([dx3, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.layer_sharp_u_5(dx4)
        sharp_out = torch.tanh(self.layer_sharp_final(dx5))

        dx1 = self.layer_vis_u_1(x8)
        dx2 = self.layer_vis_u_2(F.interpolate(torch.cat([dx1, x7], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx2 = F.interpolate(dx2, [x6.size(2), x6.size(3)], mode='bilinear', align_corners=False)
        dx3 = self.layer_vis_u_3(F.interpolate(torch.cat([dx2, x6], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = F.interpolate(dx3, [x5.size(2), x5.size(3)], mode='bilinear', align_corners=False)
        dx4 = self.layer_vis_u_4(F.interpolate(torch.cat([dx3, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.layer_vis_u_5(dx4)
        vis_out = torch.tanh(self.layer_vis_final(dx5))

        bn, _, row, col = vis_out.size()

        intensity = 0.5 * (1.01 * intensity + 1)
        intensity = torch.clamp(intensity, 0, 1)
        intensity = intensity.view(bn, self.SGNum, 3, 1, 1)

        axis = torch.clamp(1.01 * axis_out, -1, 1)
        axis = axis.view(bn, self.SGNum, 3, row, col)
        axis = axis / torch.clamp(torch.sqrt(torch.sum(axis * axis, dim=2, keepdim=True)), min=1e-6)

        sharp = 0.5 * (1.01 * sharp_out + 1)
        sharp = torch.clamp(sharp, 0, 1)
        sharp = sharp.view(bn, self.SGNum, 1, row, col)

        vis = 0.5 * (1.2 * vis_out + 1)
        vis = torch.clamp(vis, 0, 1)
        vis = vis.view(bn, self.SGNum, 1, row, col)
        return axis, sharp, intensity, vis


class BRDFContextNet(nn.Module):
    def __init__(self, cfg):
        super(BRDFContextNet, self).__init__()
        if cfg.input == 'rgb':
            in_ch = 3
        else:
            in_ch = 5
        self.layer_d_1 = make_layer(pad_type='rep', in_ch=in_ch, out_ch=64, kernel=4, stride=2, num_group=4, norm_layer=cfg.norm_layer)
        self.layer_d_2 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_6 = make_layer(in_ch=512, out_ch=1024, kernel=3, stride=1, num_group=64, norm_layer=cfg.norm_layer)

        self.layer_u_1 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_final = make_layer(pad_type='rep', in_ch=128, out_ch=cfg.dim, kernel=3, stride=1, act='None', norm_layer='None')

    @autocast()
    def forward(self, x):
        x1 = self.layer_d_1(x)
        x2 = self.layer_d_2(x1)
        x3 = self.layer_d_3(x2)
        x4 = self.layer_d_4(x3)
        x5 = self.layer_d_5(x4)
        x6 = self.layer_d_6(x5)

        dx1 = self.layer_u_1(x6)
        dx2 = self.layer_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = self.layer_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx4 = self.layer_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        x_out = self.layer_final(dx4)
        return x_out


class BRDFRefineNet(nn.Module):
    def __init__(self, cfg):
        super(BRDFRefineNet, self).__init__()
        input_ch = cfg.BRDF.aggregation.final_hidden + 5
        if cfg.BRDF.refine.context_concat:
            input_ch += cfg.BRDF.context_feature.dim
        norm_layer = cfg.BRDF.refine.norm_layer
        self.refine_d_1 = make_layer(pad_type='rep', in_ch=input_ch, out_ch=64, kernel=4, stride=2, num_group=4, norm_layer='batch')
        self.refine_d_2 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, norm_layer=norm_layer)
        self.refine_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=norm_layer)
        self.refine_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=norm_layer)
        self.refine_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=norm_layer)
        self.refine_d_6 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=norm_layer)

        self.refine_albedo_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=norm_layer)
        self.refine_albedo_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_albedo_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_albedo_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=norm_layer)
        self.refine_albedo_u_5 = make_layer(in_ch=256, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)
        self.refine_albedo_u_6 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)
        self.refine_albedo_final = make_layer(pad_type='rep', in_ch=64, out_ch=3, kernel=3, stride=1, act='None', norm_layer='None')

        self.refine_rough_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=norm_layer)
        self.refine_rough_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_rough_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_rough_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=norm_layer)
        self.refine_rough_u_5 = make_layer(in_ch=256, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)
        self.refine_rough_u_6 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)
        self.refine_rough_final = make_layer(pad_type='rep', in_ch=64, out_ch=1, kernel=3, stride=1, act='None', norm_layer='None')

    @autocast()
    def forward(self, x):
        x1 = self.refine_d_1(x)
        x2 = self.refine_d_2(x1)
        x3 = self.refine_d_3(x2)
        x4 = self.refine_d_4(x3)
        x5 = self.refine_d_5(x4)
        x6 = self.refine_d_6(x5)

        dx1 = self.refine_albedo_u_1(x6)
        dx2 = self.refine_albedo_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = self.refine_albedo_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx4 = self.refine_albedo_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.refine_albedo_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx6 = self.refine_albedo_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        albedo = self.refine_albedo_final(dx6)
        albedo = 0.5 * (torch.clamp(1.01 * torch.tanh(albedo), -1, 1) + 1)

        dx1 = self.refine_rough_u_1(x6)
        dx2 = self.refine_rough_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = self.refine_rough_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx4 = self.refine_rough_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.refine_rough_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx6 = self.refine_rough_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        rough = self.refine_rough_final(dx6)
        rough = 0.5 * (torch.clamp(1.01 * torch.tanh(rough), -1, 1) + 1)
        return albedo, rough


import math

# kernel, stride, padding
convnet = [[4, 2, 1], [4, 2, 1], [4, 2, 1], [4, 2, 1], [4, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]]
layer_names = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'final']
# convnet = [[7, 2, 3], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], ]
# layer_names = ['conv1', 'pool1', 'layer1-1', 'layer1-2', 'layer1-3', 'layer1-4', 'layer2-1', 'layer2-2', 'layer2-3', 'layer2-4', 'layer3-1', 'layer3-2', 'layer3-3', 'layer3-4', 'layer4-1', 'layer4-2', 'layer4-3', 'layer4-4',]

imsize = 640


def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t size: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))


layerInfos = []
if __name__ == '__main__':
    # first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print("------------------------")
