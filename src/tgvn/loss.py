"""
MS_SSIM_L1Loss implementation borrowed from
github.com/psyrocloud/MS-SSIM_L1_LOSS
"""

import torch
from torch import nn
from torch.nn.functional import l1_loss, conv2d


class MS_SSIM_L1Loss(nn.Module):
    def __init__(
        self,
        gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
        K=(0.01, 0.03),
        alpha=0.025,
        compensation=20.0
    ):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.compensation = compensation
        self.pad = int(2 * gaussian_sigmas[-1])
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (
                len(gaussian_sigmas), 1,
                filter_size, filter_size
            )
        )
        for idx, sigma in enumerate(gaussian_sigmas):
            g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(
                filter_size, sigma
            )

        # Register g_masks as a buffer
        self.register_buffer('g_masks', g_masks)

    def _fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y, data_range=1.0):
        C1 = (self.K[0] * data_range) ** 2
        C2 = (self.K[1] * data_range) ** 2

        mux = conv2d(x, self.g_masks, padding=self.pad)
        muy = conv2d(y, self.g_masks, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = conv2d(x * x, self.g_masks, padding=self.pad) - mux2
        sigmay2 = conv2d(y * y, self.g_masks, padding=self.pad) - muy2
        sigmaxy = conv2d(x * y, self.g_masks, padding=self.pad) - muxy

        luminance = (2 * muxy + C1) / (mux2 + muy2 + C1)
        cs = (2 * sigmaxy + C2) / (sigmax2 + sigmay2 + C2)

        lM = luminance[:, -1, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs

        loss_l1 = l1_loss(x, y, reduction='none')
        gaussian_l1 = conv2d(
            loss_l1,
            self.g_masks.narrow(dim=0, start=-1, length=1),
            padding=self.pad
        ).mean(1)

        loss_mix = (
            self.alpha * loss_ms_ssim
            + (1 - self.alpha) * gaussian_l1 / data_range
        )
        loss_mix = self.compensation * loss_mix
        return loss_mix.mean()
