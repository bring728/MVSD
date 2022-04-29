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

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch.cuda.amp.autocast_mode import autocast


def make_layer(pad_type='zeros', padding=1, in_ch=3, out_ch=64, kernel=3, stride=1, num_group=4, act='relu', normalization='group'):
    layers = []
    if pad_type == 'rep':
        layers.append(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, bias=True, padding_mode='replicate',
                      padding=padding))
    elif pad_type == 'zeros':
        layers.append(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, bias=True, padding_mode='zeros',
                      padding=padding))
    else:
        assert 'not implemented pad'

    if normalization == 'group':
        layers.append(nn.GroupNorm(num_groups=num_group, num_channels=out_ch))
    elif normalization == 'Batch':
        layers.append(nn.BatchNorm2d(out_ch))
    elif normalization == 'None':
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

        self.layer_d_1 = make_layer(pad_type='rep', in_ch=4, out_ch=64, kernel=4, stride=2, num_group=4,
                                          normalization=cfg.normalization)
        self.layer_d_2 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, normalization=cfg.normalization)
        self.layer_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, normalization=cfg.normalization)
        self.layer_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, normalization=cfg.normalization)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, normalization=cfg.normalization)
        self.layer_d_6 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, normalization=cfg.normalization)

        self.layer_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, normalization=cfg.normalization)
        self.layer_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, normalization=cfg.normalization)
        self.layer_u_3 = make_layer(in_ch=512, out_ch=256, kernel=3, stride=1, num_group=16, normalization=cfg.normalization)
        self.layer_u_4 = make_layer(in_ch=512, out_ch=128, kernel=3, stride=1, num_group=8, normalization=cfg.normalization)
        self.layer_u_5 = make_layer(in_ch=256, out_ch=64, kernel=3, stride=1, num_group=4, normalization=cfg.normalization)
        self.layer_u_6 = make_layer(in_ch=128, out_ch=64, kernel=3, stride=1, num_group=4, normalization=cfg.normalization)

        self.layer_final = make_layer(pad_type='rep', in_ch=64, out_ch=3, kernel=3, stride=1, act='None', normalization='None')

    @autocast()
    def forward(self, x):
        x1 = self.layer_d_1(x)
        x2 = self.layer_d_2(x1)
        x3 = self.layer_d_3(x2)
        x4 = self.layer_d_4(x3)
        x5 = self.layer_d_5(x4)
        x6 = self.layer_d_6(x5)

        dx1 = self.layer_u_1(x6)
        dx2 = self.layer_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear'))
        dx3 = self.layer_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear'))
        dx4 = self.layer_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear'))
        dx5 = self.layer_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear'))
        dx6 = self.layer_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear'))

        x_orig = self.layer_final(dx6)

        x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
        normal = x_orig / torch.clamp(norm, min=1e-6)
        return normal


class DirectLightingNet(nn.Module):
    def __init__(self, cfg):
        super(DirectLightingNet, self).__init__()
        self.layer_1 = make_layer(pad_type='rep', in_ch=7, out_ch=32, kernel=3, stride=1, num_group=4,
                                        normalization=cfg.normalization)
        self.layer_2 = make_layer(in_ch=32, out_ch=32, kernel=3, stride=1, num_group=4, normalization=cfg.normalization)
        self.layer_3 = make_layer(in_ch=32, out_ch=64, kernel=3, stride=1, num_group=4, normalization=cfg.normalization)

        self.layer_d_1 = make_layer(in_ch=64, out_ch=128, kernel=4, stride=2, num_group=8, normalization=cfg.normalization)
        self.layer_d_2 = make_layer(in_ch=128, out_ch=128, kernel=4, stride=2, num_group=8, normalization=cfg.normalization)
        self.layer_d_3 = make_layer(in_ch=128, out_ch=256, kernel=4, stride=2, num_group=16, normalization=cfg.normalization)
        self.layer_d_4 = make_layer(in_ch=256, out_ch=256, kernel=4, stride=2, num_group=16, normalization=cfg.normalization)
        self.layer_d_5 = make_layer(in_ch=256, out_ch=512, kernel=4, stride=2, num_group=32, normalization=cfg.normalization)
        self.layer_d_6 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, normalization=cfg.normalization)

        self.layer_u_1 = make_layer(in_ch=512, out_ch=512, kernel=3, stride=1, num_group=32, normalization=cfg.normalization)
        self.layer_u_2 = make_layer(in_ch=1024, out_ch=256, kernel=3, stride=1, num_group=16, normalization=cfg.normalization)
        self.layer_final = make_layer(pad_type='rep', in_ch=64, out_ch=cfg.SGNum * 7, kernel=3, stride=1, act='None',
                                            normalization='None')

    @autocast()
    def forward(self, x):
        xp1 = self.layer_1(x)
        xp2 = self.layer_2(xp1)
        xp3 = self.layer_3(xp2)

        x1 = self.layer_d_1(xp3)
        x2 = self.layer_d_2(x1)
        x3 = self.layer_d_3(x2)
        x4 = self.layer_d_4(x3)
        x5 = self.layer_d_5(x4)
        x6 = self.layer_d_6(x5)

        dx1 = self.layer_u_1(x6)
        dx2 = self.layer_u_2(F.interpolate(torch.cat([dx1, x5], dim=1), scale_factor=2, mode='bilinear'))
        dx3 = self.layer_u_3(F.interpolate(torch.cat([dx2, x4], dim=1), scale_factor=2, mode='bilinear'))
        dx4 = self.layer_u_4(F.interpolate(torch.cat([dx3, x3], dim=1), scale_factor=2, mode='bilinear'))
        dx5 = self.layer_u_5(F.interpolate(torch.cat([dx4, x2], dim=1), scale_factor=2, mode='bilinear'))
        dx6 = self.layer_u_6(F.interpolate(torch.cat([dx5, x1], dim=1), scale_factor=2, mode='bilinear'))

        x_orig = self.layer_final(dx6)

        x_orig = torch.clamp(1.01 * torch.tanh(x_orig), -1, 1)
        norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1)).expand_as(x_orig)
        normal = x_orig / torch.clamp(norm, min=1e-6)
        return normal
