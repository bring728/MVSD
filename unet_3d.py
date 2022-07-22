# https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
from torch import nn
import torch


# @torch.jit.script
# def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
#     """
#     Center-crops the encoder_layer to the size of the decoder_layer,
#     so that merging (concatenation) between levels/blocks is possible.
#     This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
#     """
#     if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
#         ds = encoder_layer.shape[2:]
#         es = decoder_layer.shape[2:]
#         assert ds[0] >= es[0]
#         assert ds[1] >= es[1]
#         if encoder_layer.dim() == 4:  # 2D
#             encoder_layer = encoder_layer[
#                             :,
#                             :,
#                             ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
#                             ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
#                             ]
#         elif encoder_layer.dim() == 5:  # 3D
#             assert ds[2] >= es[2]
#             encoder_layer = encoder_layer[
#                             :,
#                             :,
#                             ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
#                             ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
#                             ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
#                             ]
#     return encoder_layer, decoder_layer


def get_conv_layer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def get_up_layer(in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2, up_mode: str = 'transposed', ):
    if up_mode == 'transposed':
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def get_maxpool_layer(kernel_size: int = 2, stride: int = 2, padding: int = 0):
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif activation == 'elu':
        return nn.ELU(inplace=False)


def get_normalization(normalization: str, num_channels: int):
    if normalization == 'batch':
        return nn.BatchNorm3d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm3d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True, act: str = 'relu', norm: str = None, conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.pooling = pool
        self.normalization = norm
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = act

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, act: str = 'relu', norm: str = None, conv_mode: str = 'same', up_mode: str = 'transposed'):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.normalization = norm
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.activation = act
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, up_mode=self.up_mode)

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding, bias=True)

        # activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels, )

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        if up_layer.size(-1) != encoder_layer.size(-1):
            encoder_layer = torch.cat([torch.zeros_like(encoder_layer), encoder_layer], dim=-1)
        merged_layer = self.concat(up_layer, encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet_3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim_global = cfg.SVL.GL.global_volume_dim
        self.filters = [10 + cfg.BRDF.aggregation.brdf_feature_dim, 64, 128, 512, 512, 1024]
        self.n_blocks = 5
        self.out_ch = 8
        self.activation = 'relu'
        self.normalization = 'batch'
        self.conv_mode = 'same'
        self.up_mode = 'transposed'

        self.down_blocks = []
        self.up_blocks = []

        # create encoder path
        for i in range(self.n_blocks):
            pooling = True if i < self.n_blocks - 1 else False
            if i == 2:
                in_ch = self.filters[i] + self.dim_global
            else:
                in_ch = self.filters[i]
            out_ch = self.filters[i + 1]
            down_block = DownBlock(in_ch=in_ch, out_ch=out_ch, pool=pooling, act=self.activation, norm=self.normalization,
                                   conv_mode=self.conv_mode, )

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(self.n_blocks - 1):
            in_ch = self.filters[-(i + 1)]
            out_ch = self.filters[-(i + 2)]
            up_block = UpBlock(in_ch=in_ch, out_ch=out_ch, act=self.activation, norm=self.normalization, conv_mode=self.conv_mode,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(self.filters[1], self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x, global_vol):
        encoder_output = []
        # Encoder pathway
        for i, module in enumerate(self.down_blocks):
            if i == 2:
                x = torch.cat([torch.cat([torch.zeros_like(x), x], dim=-1), global_vol], dim=1)
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if
                      '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
