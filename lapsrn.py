import torch
import torch.nn as nn
import numpy as np
import math
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)

        self.convt_I3 = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block)

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

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2

        convt_F3 = self.convt_F3(convt_F2)
        convt_I3 = self.convt_I3(HR_4x)
        convt_R3 = self.convt_R3(convt_F3)
        HR_8x = convt_I3 + convt_R3

        return HR_2x, HR_4x, HR_8x

class Net4x(Net):
    def __init__(self):
        super(Net4x, self).__init__()

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2

        return HR_2x, HR_4x

class Net2x(Net):
    def __init__(self):
        super(Net2x, self).__init__()

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        return HR_2x


class L1CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss

class HighFrequencyLoss(nn.Module):
    def __init__(self):
        super(HighFrequencyLoss, self).__init__()
        self.log_layer = LaplacianOfGaussian(7, 1.3)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = self.log_layer(diff)
        loss = (torch.abs(log_layer(generated_img - im_post) ** 2)).mean()
        return loss

class MixedLoss(nn.Module):
    def __init__(self):
        super(MixedLoss, self).__init__()
        self.eps = 1e-6
        self.log_layer = LaplacianOfGaussian(7, 1.3)

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error_cb = torch.sqrt( diff * diff + self.eps )
        error_hfs = self.log_layer(diff)
        size = X.size()
        loss_cb = torch.sum(error_cb) / (size[0] * size[1] * size[2] * size[3])
        loss_hfs = (self.log_layer(diff) ** 2).mean()
        return loss_cb, loss_hfs
