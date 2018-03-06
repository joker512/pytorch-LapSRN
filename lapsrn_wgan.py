import torch
import torch.nn as nn
import numpy as np
import math
import sys
import scipy
import scipy.ndimage
from torch.autograd import Variable
import torch.nn.functional as F

class LaplacianOfGaussian(nn.Module):
    def __init__(self, gaussian_kernel_size, gaussian_sigma):
        super(LaplacianOfGaussian, self).__init__()
        halfkernel = int(gaussian_kernel_size / 2)
        self.reflection_pad = nn.ReflectionPad2d(halfkernel)

        kernel = np.zeros((gaussian_kernel_size, gaussian_kernel_size), dtype=np.float32)
        kernel[halfkernel, halfkernel] = 1
        self.hybrid_kernel = scipy.ndimage.filters.gaussian_laplace(kernel, gaussian_sigma)

        self.torch_hybrid_kernel = self.extend_filter(torch.from_numpy(self.hybrid_kernel), 3)
        if torch.cuda.is_available():
            self.torch_hybrid_kernel = self.torch_hybrid_kernel.cuda()

    def forward(self, x):
        h = self.reflection_pad(x)
        h = F.conv2d(h, self.torch_hybrid_kernel)
        return h

    @staticmethod
    def extend_filter(kernel, n_layers):
        rgb_kernel = torch.stack([kernel for _ in range(n_layers)])
        rgb_kernel = Variable(
            torch.stack([rgb_kernel for _ in range(n_layers)]))
        return rgb_kernel

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output


class MSLapSRN(nn.Module):
    def __init__(self):
        super(MSLapSRN, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block, True)

        self.convt_I2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block, False)

        self.convt_I3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block, True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, with_transpose = True):
        layers = []
        layers.append(block())
        if with_transpose:
            layers.append(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F1 = self.convt_F1(convt_F1)
        convt_I1 = self.convt_I1(HR_2x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_4x = convt_I1 + convt_R1

        convt_F1 = self.convt_F1(convt_F1)
        convt_I1 = self.convt_I1(HR_4x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_8x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        HR_gan = self.convt_R2(convt_F2) + HR_8x

        return HR_2x, HR_4x, HR_8x, HR_gan

class Net4x(MSLapSRN):
    def __init__(self):
        super(Net4x, self).__init__()

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F1 = self.convt_F1(convt_F1)
        convt_I1 = self.convt_I1(HR_2x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_4x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        HR_gan = self.convt_R2(convt_F2) + HR_4x

        return HR_2x, HR_4x, HR_gan

class Net2x(MSLapSRN):
    def __init__(self):
        super(Net2x, self).__init__()

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        HR_gan = self.convt_R2(convt_F2) + HR_2x

        return HR_2x, HR_gan

class _patchGan(nn.Module):
    def __init__(self, num_layers=3, average_output=False, filter_multiplier=1.0, kernel_size=3, padding='SAME'):
        super(_patchGan, self).__init__()
        self.num_layers = num_layers
        self.average_output = average_output
        self.filter_multiplier = filter_multiplier
        self.kernel_size = kernel_size
        self.padding = padding
        if self.padding == 'VALID' and self.kernel_size % 2 == 0:
            raise ValueError("You cannot use VALID padding with an even kernel size.")

        num_filters = int(self.filter_multiplier * 64)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=2*num_filters, out_channels=4*num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=4*num_filters, out_channels=8*num_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=8*num_filters, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),

            nn.AvgPool3d(kernel_size=(1, 16, 16), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.features(input)
        out = out.mean(0)

        return out.view(1)

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.features = nn.Sequential(
            # input is (3) x 256 x 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 256 x 256
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 128 x 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 64 x 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 32 x 32
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (1024) x 16 x 16
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(1024 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        out = self.features(input)

        # state size. (1024) x 8 x 8
        out = out.view(out.size(0), -1)

        # state size. (1024 x 8 x 8)
        out = self.fc1(out)

        # state size. (2048)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        #out = self.sigmoid(out)
        # state size. (1)

        out = out.mean(0)
        return out.view(1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.netg = MSLapSRN()
        self.netd = _patchGan()

    def forward(self, input):
        HR_2x, HR_4x, HR_8x, HR_gan = self.netg(input)
        out = self.netd(HR_gan)
        return out, HR_2x, HR_4x, HR_8x, HR_gan


class L1Loss(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        size = X.size()
        loss = torch.sum(diff) / (size[0] * size[1] * size[2] * size[3])
        return loss

class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        size = X.size()
        loss = torch.sum(error) / (size[0] * size[1] * size[2] * size[3])
        return loss

class HighFrequencyLoss(nn.Module):
    def __init__(self, kernel=5, sigma=1.3):
        super(HighFrequencyLoss, self).__init__()
        self.log_layer = LaplacianOfGaussian(kernel, sigma)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = self.log_layer(diff)
        loss = (error ** 2).mean()
        return loss

class MixedLoss(nn.Module):
    def __init__(self, cb_eps=1e-6, hfs_kernel=5, hfs_sigma=1.3):
        super(MixedLoss, self).__init__()
        self.eps = cb_eps
        self.log_layer = LaplacianOfGaussian(hfs_kernel, hfs_sigma)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error_cb = torch.sqrt( diff * diff + self.eps )
        error_hfs = torch.add(self.log_layer(X), -self.log_layer(Y))
        size = X.size()
        loss_cb = torch.sum(error_cb) / (size[0] * size[1] * size[2] * size[3])
        loss_hfs = (error_hfs ** 2).mean()
        return loss_cb, loss_hfs
