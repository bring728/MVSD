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
import cv2
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from utils import *
from einops import rearrange


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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
        filters = [64, 128, 256]

        # original
        layers = [3, 4, 6]
        norm_layer = nn.BatchNorm2d
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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, cfg.dim, 3, 1)

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

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)
        return x_out


def make_layer(pad_type='zeros', padding=1, in_ch=3, out_ch=64, kernel=3, stride=1, num_group=4, act='relu', norm_layer='group'):
    layers = []
    if pad_type == 'rep':
        layers.append(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, bias=norm_layer != 'Batch',
                      padding_mode='replicate', padding=padding))
    elif pad_type == 'zeros':
        layers.append(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, bias=norm_layer != 'Batch',
                      padding_mode='zeros', padding=padding))
    else:
        assert 'not implemented pad'

    if norm_layer == 'group':
        layers.append(nn.GroupNorm(num_groups=num_group, num_channels=out_ch))
    elif norm_layer == 'batch':
        layers.append(nn.BatchNorm2d(out_ch))
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
        filters = [32, 64, 128, 256, 256, 512, 512, 1024]
        color_feat_dim = 256
        remain_feat_dim = filters[7] - color_feat_dim
        self.SGNum = cfg.SGNum

        self.layer_d_1 = make_layer(pad_type='rep', in_ch=8, out_ch=filters[0], kernel=4, stride=2, num_group=2, norm_layer=cfg.norm_layer)
        self.layer_d_2 = make_layer(in_ch=32, out_ch=filters[1], kernel=4, stride=2, num_group=4, norm_layer=cfg.norm_layer)
        self.layer_d_3 = make_layer(in_ch=64, out_ch=filters[2], kernel=4, stride=2, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_d_4 = make_layer(in_ch=128, out_ch=filters[3], kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=filters[4], kernel=4, stride=2, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_d_6 = make_layer(in_ch=256, out_ch=filters[5], kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_7 = make_layer(in_ch=512, out_ch=filters[6], kernel=4, stride=2, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_d_8 = make_layer(in_ch=512, out_ch=filters[7], kernel=3, stride=1, num_group=64, norm_layer=cfg.norm_layer)

        activation_func = nn.ELU(inplace=True)
        self.layer_intensity = nn.Sequential(nn.Linear(color_feat_dim, 128), activation_func,
                                             nn.Linear(128, 64), activation_func,
                                             nn.Linear(64, 32), activation_func,
                                             nn.Linear(32, cfg.SGNum * 3), nn.Tanh(),
                                             )

        self.layer_u_1 = make_layer(in_ch=768, out_ch=768, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_u_2 = make_layer(in_ch=1280, out_ch=768, kernel=3, stride=1, num_group=32, norm_layer=cfg.norm_layer)
        self.layer_u_3 = make_layer(in_ch=1280, out_ch=768, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_4 = make_layer(in_ch=1024, out_ch=512, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_5 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_6 = make_layer(in_ch=256, out_ch=128, kernel=3, stride=1, num_group=16, norm_layer=cfg.norm_layer)
        self.layer_u_7 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=8, norm_layer=cfg.norm_layer)
        self.layer_final = make_layer(pad_type='rep', in_ch=64, out_ch=cfg.SGNum * 5, kernel=3, stride=1, act='None', norm_layer='None')

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
        x_color = F.adaptive_avg_pool2d(x8[:, :256, ...], (1, 1))[..., 0, 0]
        intensity = 1.01 * self.layer_intensity(x_color)

        dx1 = self.layer_u_1(x8[:, 256:, ...])
        dx2 = self.layer_u_2(F.interpolate(torch.cat([dx1, x7], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx2 = F.interpolate(dx2, [x6.size(2), x6.size(3)], mode='bilinear', align_corners=False)
        dx3 = self.layer_u_3(F.interpolate(torch.cat([dx2, x6], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = F.interpolate(dx3, [x5.size(2), x5.size(3)], mode='bilinear', align_corners=False)
        dx4 = self.layer_u_4(F.interpolate(torch.cat([dx3, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.layer_u_5(dx4)
        dx6 = self.layer_u_6(dx5)
        dx7 = self.layer_u_7(dx6)
        x_out = torch.tanh(self.layer_final(dx7))
        bn, _, row, col = x_out.size()

        sharp = 1.01 * x_out[:, :self.SGNum]
        sharp = 0.5 * (sharp + 1)
        sharp = torch.clamp(sharp, 0, 1)
        sharp = sharp.view(bn, self.SGNum, 1, row, col)

        vis = 1.2 * x_out[:, self.SGNum:self.SGNum * 2]
        vis = 0.5 * (vis + 1)
        vis = torch.clamp(vis, 0, 1)
        vis = vis.view(bn, self.SGNum, 1, row, col)

        axis = 1.01 * x_out[:, self.SGNum * 2:]
        axis = axis.view(bn, self.SGNum, 3, row, col)
        axis = axis / torch.clamp(torch.sqrt(torch.sum(axis * axis, dim=2, keepdim=True)), min=1e-6)

        intensity = 0.5 * (intensity + 1)
        intensity = torch.clamp(intensity, 0, 1)
        intensity = intensity.view(bn, self.SGNum, 3, 1, 1)
        return axis, sharp, intensity, vis


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'B H W n (h d) -> B H W h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots * kwargs['mask']
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'B H W h n d -> B H W n (h d)')
        return self.to_out(out)


class Visible_Attention(nn.Module):
    def __init__(self, dim, dim_head=64, dropout=0.):
        super().__init__()
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_out = nn.Linear(dim_head, dim)

    def forward(self, x, **kwargs):
        v = self.to_v(x)
        out = torch.matmul(kwargs['mask'], v)
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_hidden, final_hidden, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Visible_Attention(dim, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_hidden, dropout=dropout))
            ]))

        self.mlp_final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, final_hidden),
            nn.GELU(),
            nn.Linear(final_hidden, final_hidden),
            nn.GELU(),
            nn.Linear(final_hidden, final_hidden),
            nn.ReLU()
        )

    def forward(self, x, weight):
        b, h, w, n, _ = x.shape
        weight = weight.transpose(-1, -2).expand(-1, -1, -1, n, -1)
        for attn, ff in self.layers:
            x = attn(x, mask=weight) + x
            x = ff(x) + x

        x = x[..., 0, :]
        x = self.mlp_final(x)
        return x


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def fused_mean_variance(x, weight, dim):
    mean = torch.sum(x * weight, dim=dim, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=dim, keepdim=True)
    return mean, var


def fused_mean(x, weight, dim):
    mean = torch.sum(x * weight, dim=dim, keepdim=True)
    return mean


class MultiViewAggregation(nn.Module):
    def __init__(self, cfg):
        super(MultiViewAggregation, self).__init__()
        self.cfg = cfg
        self.activation = nn.ELU(inplace=True)
        if cfg.BRDF.aggregation.input == 'vector':
            input_ch = 3 + 3 + cfg.DL.SGNum * 7
        else:
            input_ch = 1 + cfg.DL.SGNum * 6
        if cfg.BRDF.refine.use:
            output_ch = 64
        else:
            output_ch = 4
        self.net_type = cfg.BRDF.aggregation.type
        hidden = cfg.BRDF.aggregation.pbr_hidden
        self.pbr_mlp = nn.Sequential(nn.Linear(input_ch, hidden), self.activation,
                                     nn.Linear(hidden, hidden), self.activation,
                                     nn.Linear(hidden, hidden), self.activation,
                                     nn.Linear(hidden, cfg.BRDF.aggregation.pbr_feature_dim), self.activation, )
        self.pbr_mlp.apply(weights_init)
        if self.net_type == 'mlp':
            self.perview_mlp = nn.Sequential(nn.Linear((cfg.BRDF.feature.dim + 3) * 3 + hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, output_ch))
            self.perview_mlp.apply(weights_init)
        elif self.net_type == 'transformer':
            input_ch = cfg.BRDF.feature.dim + 3 + cfg.BRDF.aggregation.pbr_feature_dim
            self.transformer = Transformer(input_ch, cfg.BRDF.aggregation.num_depth, cfg.BRDF.aggregation.num_head,
                                           cfg.BRDF.aggregation.head_dim, cfg.BRDF.aggregation.mlp_hidden, cfg.BRDF.aggregation.final_hidden)

    def forward(self, rgb_feat, view_dir, proj_err, normal, DL):
        bn, h, w, num_views, _ = rgb_feat.shape
        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
        weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + TINY_NUMBER)

        DL = DL.reshape(bn, h, w, 1, self.cfg.DL.SGNum, 7)
        axis = DL[..., :3]
        DL_axis = axis / torch.clamp(torch.sqrt(torch.sum(axis * axis, dim=-1, keepdim=True)), min=1e-6)
        if self.cfg.BRDF.aggregation.input == 'vector':
            DL[..., :3] = DL_axis
            DL = DL.reshape((bn, h, w, 1, -1))
            pbr_batch = torch.cat([torch.cat([normal, DL], dim=-1).expand(-1, -1, -1, num_views, -1), view_dir], dim=-1)
        else:
            DLdotN = torch.sum(DL_axis * normal[:, :, :, :, None], dim=-1)
            hdotV = (torch.sum(DL_axis * view_dir[:, :, :, :, None], dim=-1) + 1) / 2
            VdotN = torch.sum(view_dir * normal, dim=-1, keepdim=True)
            sharp = DL[..., 3:4].reshape((bn, h, w, 1, -1))
            intensity = DL[..., 4:].reshape((bn, h, w, 1, -1))
            pbr_batch = torch.cat([torch.cat([DLdotN, sharp, intensity], dim=-1).expand(-1, -1, -1, num_views, -1), hdotV, VdotN], dim=-1)
        pbr_feature = self.pbr_mlp(pbr_batch)

        if self.net_type == 'mlp':
            mean, var = fused_mean_variance(rgb_feat, weight, dim=-2)
            globalfeat = torch.cat([mean, var], dim=-1)
            x = torch.cat([globalfeat.expand(-1, -1, -1, num_views, -1), rgb_feat, pbr_feature], dim=-1)
            x = self.perview_mlp(x)
            x = fused_mean(x, weight, dim=-2).squeeze(-2)
            if self.cfg.BRDF.refine.use:
                x = self.activation(x)
            else:
                x = 0.5 * (torch.clamp(1.01 * torch.tanh(x), -1, 1) + 1)
            # view_dir_var = torch.var(torch.cat([torch.arctan(view_dir[..., 1:2] / view_dir[..., :1]), torch.arccos(view_dir[..., 2:])], dim=-1),
            #                          dim=-2, unbiased=False)
            # weight_var = torch.var(weight, dim=-2, unbiased=False)
            # return torch.cat([x, view_dir_var, weight_var], dim=-1)
        else:
            rgb_feat_pbr = torch.cat([rgb_feat, pbr_feature], dim=-1)
            x = self.transformer(rgb_feat_pbr, weight)
            x = 0.5 * (torch.clamp(1.01 * torch.tanh(x), -1, 1) + 1)
        return x


class BRDFRefineNet(nn.Module):
    def __init__(self, cfg):
        super(BRDFRefineNet, self).__init__()
        input_ch = cfg.BRDF.aggregation.final_hidden + 5
        norm_layer = cfg.BRDF.refine.norm_layer
        self.refine_d_1 = make_layer(pad_type='rep', in_ch=input_ch, out_ch=64, kernel=4, stride=2, num_group=4, norm_layer=norm_layer)
        self.refine_d_2 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, norm_layer=norm_layer)
        self.refine_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=norm_layer)
        self.refine_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, norm_layer=norm_layer)
        self.refine_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, norm_layer=norm_layer)
        self.refine_d_6 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=norm_layer)

        self.refine_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, norm_layer=norm_layer)
        self.refine_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, norm_layer=norm_layer)
        self.refine_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, norm_layer=norm_layer)
        self.refine_u_5 = make_layer(in_ch=256, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)
        self.refine_u_6 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=4, norm_layer=norm_layer)

        if cfg.BRDF.conf.use:
            self.refine_final = make_layer(pad_type='rep', in_ch=64, out_ch=5, kernel=3, stride=1, act='None', norm_layer='None')
        else:
            self.refine_final = make_layer(pad_type='rep', in_ch=64, out_ch=4, kernel=3, stride=1, act='None', norm_layer='None')

    @autocast()
    def forward(self, x):
        x1 = self.refine_d_1(x)
        x2 = self.refine_d_2(x1)
        x3 = self.refine_d_3(x2)
        x4 = self.refine_d_4(x3)
        x5 = self.refine_d_5(x4)
        x6 = self.refine_d_6(x5)

        dx1 = self.refine_u_1(x6)
        dx2 = self.refine_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx3 = self.refine_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx4 = self.refine_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx5 = self.refine_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        dx6 = self.refine_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=False))
        x_out = self.refine_final(dx6)

        x_out = 0.5 * (torch.clamp(1.01 * torch.tanh(x_out), -1, 1) + 1)
        return x_out


import math

# kernel, stride, padding
convnet = [[4, 2, 1], [4, 2, 1], [4, 2, 1], [4, 2, 1], [4, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]]
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

