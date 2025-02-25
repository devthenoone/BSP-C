from _py_abc import ABCMeta
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, Sequence

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from networks.blocks import ConvBnReLU, ResBlock, conv3d, Attention


class ConcatHNN1FC(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4, ndim_non_img=10) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, 32)
        self.fc1 = nn.Linear(32, n_outputs)


    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)
        out = self.fc1(out)
        return out


class ConcatHNN2FC(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4, ndim_non_img=10, bottleneck_dim=15) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, bottleneck_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(bottleneck_dim, n_outputs)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)

        return out


class HeterogeneousResNet(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)


    def forward(self, image):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class InteractiveHNN(nn.Module):

    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4, ndim_non_img=10) -> None:
        super().__init__()

        # ResNet
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 15, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(15, n_basefilters, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

        self.aux_2 = nn.Linear(n_basefilters, n_basefilters, bias=False)
        self.aux_3 = nn.Linear(n_basefilters, 2 * n_basefilters, bias=False)
        self.aux_4 = nn.Linear(2 * n_basefilters, 4 * n_basefilters, bias=False)


    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        tabular = tabular.to(torch.float32)

        attention = self.aux(tabular)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block1(out)

        attention = self.aux_2(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block2(out)

        attention = self.aux_3(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block3(out)

        attention = self.aux_4(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block4(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out1 = F.softmax(out, dim=1)

        return out1

class DAFT2021(nn.Module):
    def __init__(
            self,
            in_channels: int=1,
            n_outputs: int=3,
            bn_momentum: float = 0.1,
            n_basefilters: int = 4,
            filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)



    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class FilmBase(nn.Module, metaclass=ABCMeta):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float,
        stride: int,
        ndim_non_img: int,
        location: int,
        activation: str,
        scale: bool,
        shift: bool,
    ) -> None:

        super().__init__()

        # sanity checks
        if location not in set(range(5)):
            raise ValueError(f"Invalid location specified: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid location specified: {location}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, affine=(location != 3))
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None
        # Film-specific variables
        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError("This is equivalent to location=1 and no downsampling!")
        # location decoding
        self.film_dims = 0
        if location in {0, 1, 2}:
            self.film_dims = in_channels
        elif location in {3, 4}:
            self.film_dims = out_channels
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None


    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""

    def forward(self, feature_map, x_aux):

        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)
        residual = feature_map

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 2:
            feature_map = self.rescale_features(feature_map, x_aux)
        out = self.conv1(feature_map)
        out = self.bn1(out)

        if self.location == 3:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out


class DAFTBlock(FilmBase):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=3,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 10,
        location: int = 3,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 15,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class Fusion2022(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4, ndim_non_img=10) -> None:
        super().__init__()


        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = FBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, 16)
        self.fc1 = nn.Linear(16, n_outputs)

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 7, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(7, 4, bias=False)),
        ]

        layers2 = [
            ("aux_base", nn.Linear(ndim_non_img, 15, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(15, 32, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.aux1 = nn.Sequential(OrderedDict(layers2))



    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)

        return out

#
class FBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1, n_basefilters=4, ndim_non_img=10):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_pool1 = nn.AvgPool3d(5)
        self.fc5 = nn.Linear(32, 4)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 7, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(7, n_basefilters, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, x, tabular):
        residual = x
        out = self.conv1(x)
        out2 = self.bn1(out)
        batch_size, n_channels = out2.size()[:2]
        attention = self.aux(tabular)
        cat2 = self.fc5(out2)
        out = torch.cat(attention, cat2)
        out = Attention(out)

        out2 = self.relu(out)
        out = self.conv2(out2)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


