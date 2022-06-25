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

class Transformer(nn.Module):
    def __init__(self, norm_layer, dim, dim_head, mlp_hidden, final_hidden):
        super().__init__()
        self.norm = nn.LayerNorm
        self.norm_1_1 = self.norm(dim)
        self.to_v_1 = nn.Linear(dim, dim_head, bias=False)
        self.to_out_1 = nn.Linear(dim_head * 2, dim)

        self.norm_1_2 = self.norm(dim)
        self.net_1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        self.norm_2_1 = self.norm(dim)
        self.to_v_2 = nn.Linear(dim, dim_head, bias=False)
        self.to_out_2 = nn.Linear(dim_head * 2, dim)

        self.norm_2_2 = self.norm(dim)
        self.net_2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        self.mlp_final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, final_hidden),
            nn.GELU(),
            nn.Linear(final_hidden, final_hidden),
            nn.GELU(),
            nn.Linear(final_hidden, final_hidden),
        )

    def forward(self, x_input, weight):
        b, h, w, n, _ = x_input.shape
        v = self.to_v_1(self.norm_1_1(x_input))
        mean, var = fused_mean_variance(v, weight, dim=-2)
        x = self.to_out_1(torch.cat([mean, var], dim=-1)).expand(-1, -1, -1, n, -1)
        x = x + x_input

        x = self.net_1(self.norm_1_2(x))
        x = x + x_input

        v = self.to_v_2(self.norm_2_1(x))
        mean, var = fused_mean_variance(v, weight, dim=-2)
        x = self.to_out_2(torch.cat([mean, var], dim=-1))[..., 0, :]
        x = x + x_input[..., 0, :]

        x = self.net_2(self.norm_2_2(x))
        x = x + x_input[..., 0, :]

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
                                     nn.Linear(hidden, cfg.BRDF.aggregation.pbr_feature_dim))
        self.pbr_mlp.apply(weights_init)
        if self.net_type == 'mlp':
            self.perview_mlp = nn.Sequential(nn.Linear((cfg.BRDF.context_feature.dim + 3) * 3 + hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, hidden), self.activation,
                                             nn.Linear(hidden, output_ch))
            self.perview_mlp.apply(weights_init)
        elif self.net_type == 'transformer':
            input_ch = 3 + cfg.BRDF.aggregation.pbr_feature_dim
            # input_ch = cfg.BRDF.context_feature.dim + 3 + cfg.BRDF.aggregation.pbr_feature_dim
            self.transformer = Transformer(cfg.BRDF.aggregation.norm_layer, input_ch, cfg.BRDF.aggregation.head_dim,
                                           cfg.BRDF.aggregation.mlp_hidden, cfg.BRDF.aggregation.final_hidden)

    def forward(self, rgb_feat, view_dir, proj_err, normal, DL):
        bn, h, w, num_views, _ = rgb_feat.shape
        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
        weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + TINY_NUMBER)

        DL = DL.reshape(bn, h, w, 1, self.cfg.DL.SGNum, 7)
        # axis = DL[..., :3]
        # DL_axis = axis / torch.clamp(torch.sqrt(torch.sum(axis * axis, dim=-1, keepdim=True)), min=1e-6)
        if self.cfg.BRDF.aggregation.input == 'vector':
            # DL[..., :3] = DL_axis
            DL = DL.reshape((bn, h, w, 1, -1))
            pbr_batch = torch.cat([torch.cat([normal, DL], dim=-1).expand(-1, -1, -1, num_views, -1), view_dir], dim=-1)
        else:
            DLdotN = torch.sum(DL[..., :3] * normal[:, :, :, :, None], dim=-1)
            hdotV = (torch.sum(DL[..., :3] * view_dir[:, :, :, :, None], dim=-1) + 1) / 2
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
            # if self.cfg.BRDF.refine.use:
            #     x = self.activation(x)
            # else:
            #     x = 0.5 * (torch.clamp(1.01 * torch.tanh(x), -1, 1) + 1)
        return x