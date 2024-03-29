# The MIT License (MIT)
# =====================
#
# Copyright © 2019-2020 Azavea
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import torch
from typing import Optional


class LearnedIndices(torch.nn.Module):

    output_channels = 32

    def __init__(self, band_count):
        super(LearnedIndices, self).__init__()
        intermediate_channels1 = 64
        kernel_size = 1
        padding_size = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(band_count,
                                     intermediate_channels1,
                                     kernel_size=kernel_size,
                                     padding=padding_size,
                                     bias=False)
        self.conv_numerator = torch.nn.Conv2d(intermediate_channels1,
                                              self.output_channels,
                                              kernel_size=1,
                                              padding=0,
                                              bias=False)
        self.conv_denominator = torch.nn.Conv2d(intermediate_channels1,
                                                self.output_channels,
                                                kernel_size=1,
                                                padding=0,
                                                bias=True)
        self.batch_norm_quotient = torch.nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        x = self.conv1(x)
        numerator = self.conv_numerator(x)
        denomenator = self.conv_denominator(x)
        x = numerator / (denomenator + 1e-7)
        x = self.batch_norm_quotient(x)
        return x


class Nugget(torch.nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Nugget, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class CheapLab(torch.nn.Module):
    def __init__(self,
                 num_channels: int,
                 preshrink: int = 1,
                 out_channels: int = 2):
        super(CheapLab, self).__init__()

        self.preshrink = preshrink
        self.indices = LearnedIndices(num_channels)
        self.classifier = torch.nn.Sequential(
            Nugget(1, self.indices.output_channels + num_channels, 16),
            Nugget(1, 16, 8),
            Nugget(1, 8, 4),
            Nugget(1, 4, out_channels),
        )

    def forward(self, x: torch.Tensor):
        [w, h] = x.shape[-2:]

        if self.preshrink != 1:
            x = torch.nn.functional.interpolate(
                x,
                size=[w // self.preshrink, h // self.preshrink],
                mode='bilinear',
                align_corners=False)
        x = torch.cat([self.indices(x), x], axis=1)
        x = self.classifier(x)
        if self.preshrink != 1:
            x = torch.nn.functional.interpolate(x,
                                                size=[w, h],
                                                mode='bilinear',
                                                align_corners=False)

        return x


class RvBCELoss(torch.nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super(RvBCELoss, self).__init__()
        self.crit = torch.nn.BCEWithLogitsLoss(weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bg = x[:, 0, :, :]
        fg = x[:, 1, :, :]
        x = fg - bg
        y = (y == 1).type(x.type())
        return self.crit(x, y)


def make_cheaplab_model(num_channels, preshrink=1, out_channels=2):
    cheaplab = CheapLab(num_channels, preshrink, out_channels)
    return cheaplab


def make_bce_loss(weight: Optional[torch.Tensor] = None):
    return RvBCELoss(weight)
