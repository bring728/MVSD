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
    def __init__(self, dim, dim_head, mlp_hidden, final_hidden, output_dim):
        super().__init__()
        self.to_v_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head, bias=False)
        )
        self.to_out_1 = nn.Linear(dim_head * 3, dim)

        self.net_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        self.to_v_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_head, bias=False)
        )
        self.to_out_2 = nn.Linear(dim_head * 3, dim)

        self.net_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        self.mlp_brdf = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, final_hidden), nn.GELU(),
            nn.Linear(final_hidden, final_hidden), nn.GELU(),
            nn.Linear(final_hidden, output_dim),
        )

    @autocast()
    def forward(self, x_input, weight):
        b, h, w, n, _ = x_input.shape
        v = self.to_v_1(x_input)
        mean, var = fused_mean_variance(v, weight, dim=-2)
        x = self.to_out_1(torch.cat([v, torch.cat([mean, var], dim=-1).expand(-1, -1, -1, n, -1)], dim=-1))
        x = x + x_input
        x = self.net_1(x)
        x = x + x_input

        v = self.to_v_2(x)
        mean, var = fused_mean_variance(v, weight, dim=-2)
        x = self.to_out_2(torch.cat([v[..., :1, :], mean, var], dim=-1))[..., 0, :]
        x = x + x_input[..., 0, :]
        x = self.net_2(x)
        x = x + x_input[..., 0, :]

        brdf_feature = self.mlp_brdf(x)
        return brdf_feature


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
        input_ch = 8 # intensity, sharp, fresnel, dot, dot, dot
        hidden = cfg.BRDF.aggregation.pbr_hidden
        self.pbr_mlp = nn.Sequential(nn.LayerNorm(input_ch),
                                     nn.Linear(input_ch, hidden), self.activation,
                                     nn.Linear(hidden, hidden), self.activation,
                                     nn.Linear(hidden, cfg.BRDF.aggregation.pbr_feature_dim))

        input_ch = cfg.BRDF.context_feature.dim + 3 + cfg.BRDF.aggregation.pbr_feature_dim
        self.transformer = Transformer(input_ch, cfg.BRDF.aggregation.head_dim,
                                       cfg.BRDF.aggregation.mlp_hidden, cfg.BRDF.aggregation.final_hidden, cfg.BRDF.aggregation.brdf_feature_dim)

    @autocast()
    def forward(self, rgb, featmaps_dense, view_dir, proj_err, normal, DL):
        bn, h, w, num_views, _ = rgb.shape
        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
        weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + TINY_NUMBER)

        DL = DL.reshape(bn, h, w, 1, self.cfg.DL.SGNum, 7).expand(-1, -1, -1, num_views, -1, -1)
        DL_axis = DL[..., :3]
        DL_sharp = 1 / (DL[..., 3:4] + 1)
        DL_intensity = DL[..., 4:]

        h = DL_axis + view_dir
        h = h / torch.sqrt(torch.clamp(torch.sum(h * h, dim=-1, keepdim=True), min=1e-6))
        NdotL = torch.clamp(torch.sum(DL_axis * normal, dim=-1, keepdim=True), min=0.0)
        NdotV = torch.sum(view_dir * normal, dim=-1, keepdim=True).expand(-1, -1, -1, -1, self.cfg.DL.SGNum, -1)
        NdotH_2 = torch.pow(torch.sum(normal * h, dim=-1, keepdim=True), 2.0)
        hdotV = torch.sum(h * view_dir, dim=-1, keepdim=True)
        fresnel = 0.95 * torch.pow(2.0, (-5.55472 * hdotV - 6.98316) * hdotV) + 0.05

        pbr_batch = torch.cat([NdotL, DL_sharp, DL_intensity, NdotH_2, NdotV, fresnel], dim=-1)
        pbr_feature_perSG = self.pbr_mlp(pbr_batch)

        pbr_mask = NdotL * torch.sum(DL_intensity, dim=-1, keepdim=True)
        notDark = (pbr_mask > 0.001).float()
        pbr_feature = torch.sum(pbr_feature_perSG * notDark, dim=-2)
        rgb_feat_pbr = torch.cat([rgb, featmaps_dense.expand(-1, -1, -1, num_views, -1), pbr_feature], dim=-1)
        brdf_feature = self.transformer(rgb_feat_pbr, weight)
        return brdf_feature
