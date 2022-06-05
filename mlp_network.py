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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import TINY_NUMBER


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


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


class BRDFNet(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(BRDFNet, self).__init__()
        activation_func = nn.ELU(inplace=True)
        if cfg.BRDF.input_pbr_mlp == 'vector':
            input_ch = 3 + 3 + cfg.DL.SGNum * 7
        elif cfg.BRDF.input_pbr_mlp == 'dot':
            input_ch = 1 + cfg.DL.SGNum * 6
        else:
            raise Exception('input_pbr_mlp error')

        self.pbr_mlp = nn.Sequential(nn.Linear(input_ch, 64), activation_func,
                                     nn.Linear(64, 64), activation_func,
                                     nn.Linear(64, 64), activation_func,
                                     nn.Linear(64, 32), activation_func, )

        self.perview_mlp = nn.Sequential(nn.Linear((cfg.BRDF.feature_dims + 3) * 3 + 32, 64), activation_func,
                                         nn.Linear(64, 64), activation_func,
                                         nn.Linear(64, 64), activation_func,
                                         nn.Linear(64, 64), activation_func,
                                         nn.Linear(64, 32), activation_func, )

        self.perview_mlp.apply(weights_init)
        self.pbr_mlp.apply(weights_init)

    def forward(self, rgb_feat, view_dir, proj_err, normal, DL):
        num_views = rgb_feat.shape[2]
        # GT weight
        weight = -torch.clamp(torch.log10(torch.abs(proj_err) + TINY_NUMBER), min=None, max=0)
        weight = weight / (torch.sum(weight, dim=1, keepdim=True) + TINY_NUMBER)

        h, w, _, n = DL.shape
        DL = DL.reshape(h, w, 1, self.cfg.DL.SGNum, 7)
        axis = DL[..., :3]
        DL_axis = axis / torch.clamp(torch.sqrt(torch.sum(axis * axis, dim=-1, keepdim=True)), min=1e-6)
        if self.cfg.BRDF.input_pbr_mlp == 'vector':
            DL[..., :3] = DL_axis
            DL = DL.reshape((h, w, 1, -1))
            pbr_batch = torch.cat([torch.cat([normal, DL], dim=-1).expand(-1, -1, num_views, -1), view_dir], dim=-1)
        else:
            DLdotN = torch.sum(DL_axis * normal[:, :, :, None], dim=-1)
            hdotV = (torch.sum(DL_axis * view_dir[:, :, :, None], dim=-1) + 1) / 2
            VdotN = torch.sum(view_dir * normal, dim=-1, keepdim=True)
            sharp = DL[..., 3:4].reshape((h, w, 1, -1))
            intensity = DL[..., 4:].reshape((h, w, 1, -1))
            pbr_batch = torch.cat([torch.cat([DLdotN, sharp, intensity], dim=-1).expand(h, w, num_views, -1), hdotV, VdotN], dim=-1)
        pbr_feature = self.pbr_mlp(pbr_batch)

        mean, var = fused_mean_variance(rgb_feat, weight, dim=-2)
        globalfeat = torch.cat([mean, var], dim=-1)
        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat, pbr_feature], dim=-1)
        x = self.perview_mlp(x)

        mean = fused_mean(x, weight, dim=-2)
        # globalfeat = torch.cat([mean.squeeze(1), var.squeeze(1), weight.mean(dim=1)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        # out = self.multiview_mlp(globalfeat)
        # return 0.5 * (torch.clamp(1.01 * out, -1, 1) + 1)
        return mean.squeeze(2)
