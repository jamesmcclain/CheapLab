# The MIT License (MIT)
# =====================
#
# Copyright © 2019 Azavea
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


class LearnedIndices(torch.nn.Module):

    output_channels = 32

    def __init__(self, band_count):
        super(LearnedIndices, self).__init__()
        intermediate_channels1 = 64
        kernel_size = 1
        padding_size = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(
            band_count, intermediate_channels1, kernel_size=kernel_size, padding=padding_size, bias=False)
        self.conv_numerator = torch.nn.Conv2d(
            intermediate_channels1, self.output_channels, kernel_size=1, padding=0, bias=False)
        self.conv_denominator = torch.nn.Conv2d(
            intermediate_channels1, self.output_channels, kernel_size=1, padding=0, bias=True)
        self.batch_norm_quotient = torch.nn.BatchNorm2d(
            self.output_channels)

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
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class CheapLabBinary(torch.nn.Module):
    def __init__(self, band_count):
        super(CheapLabBinary, self).__init__()
        self.indices = LearnedIndices(band_count)
        self.classifier = torch.nn.Sequential(
            Nugget(1, self.indices.output_channels+band_count, 16),
            Nugget(1, 16, 8),
            Nugget(1, 8, 4),
            Nugget(1, 4, 2),
            torch.nn.Conv2d(2, 1, kernel_size=1)
        )
        self.input_layers = [self.indices]
        self.output_layers = [self.classifier]

    def forward(self, x):
        x = torch.cat([self.indices(x), x], axis=1)
        x = self.classifier(x)
        return {'2seg': x}


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    cheaplab = CheapLabBinary(band_count)
    return cheaplab
