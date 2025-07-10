"""
Copyright (c) 2025, New York University
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmri import complex_abs, complex_conj, ifft2c, fft2c, rss_complex
from fastmri.models.unet import Unet


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fft2c(complex_mul(x, sens_maps))


def sens_reduce(k: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul(
        ifft2c(k), complex_conj(sens_maps)
    ).sum(dim=1, keepdim=True)


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    x_re, x_im = x[..., 0].clone(), x[..., 1].clone()
    y_re, y_im = y[..., 0].clone(), y[..., 1].clone()

    re = x_re * y_re - x_im * y_im
    im = x_re * y_im + x_im * y_re
    return torch.stack((re, im), dim=-1)


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Args:
        a, b are B x 1 x H x W x 2 tensors
        where the last dimension represents
        the real and imaginary parts
    """
    B, C, H, W, two = a.shape
    assert (two == 2) and (a.shape == b.shape)
    a = a.view(B, C, H * W, 2)
    b = b.view(B, C, H * W, 2)
    re_a, im_a = a[..., 0].clone(), a[..., 1].clone()
    re_b, im_b = b[..., 0].clone(), b[..., 1].clone()
    out = re_a * re_b + im_a * im_b
    out = out.sum(dim=2, keepdim=True).unsqueeze(2)
    return out


class NormUnet(nn.Module):
    """
    Adapted from the fastMRI implementation.
    fastMRI is copyright-protected by Facebook, Inc. and its affiliates.

    Same as a regular U-Net, but with normalization applied to the input
    before the U-Net. This keeps the values more numerically stable.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    # Normalize the x as if it was complex valued, modified from fastMRI repo
    def norm(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply Cholesky whitening
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)

        mean = torch.mean(x, dim=2, keepdim=True)
        x = x - mean  # x is now 0 mean
        xT = x.permute(0, 2, 1)
        cov = torch.bmm(x, xT) / (c // 2 * h * w - 1)
        L = torch.linalg.cholesky(cov)
        W = torch.linalg.inv(L)
        x = torch.bmm(W, x).contiguous().view(b, c, h, w)
        return x, mean, L

    # Unnormalize x as if it was complex valued, modified from fastMRI repo
    def unnorm(self, x, mean, L):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        x = torch.bmm(L, x) + mean
        x = x.contiguous().view(b, c, h, w)
        return x

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, L = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, L)
        x = self.chan_complex_to_last_dim(x)
        return x


class SensitivityModel(nn.Module):
    """
    Adapted from the fastMRI implementation.
    fastMRI is copyright-protected by Facebook, Inc. and its affiliates.

    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(
        self,
        masked_kspace: torch.Tensor,
    ) -> torch.Tensor:
        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class VarNetBlockImage(nn.Module):
    """
    Adapted from the fastMRI implementation.
    fastMRI is copyright-protected by Facebook, Inc. and its affiliates.

    Model block for end-to-end variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_image: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:

        current_kspace = sens_expand(current_image, sens_maps)
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero)
        soft_dc *= torch.abs(self.dc_weight)
        soft_dc = sens_reduce(soft_dc, sens_maps)
        model_term = self.model(current_image)
        return current_image - soft_dc - model_term


class TGVN_Block(nn.Module):
    """
    Model block for trust-guided variational network.
    A series of these blocks can be stacked to form
    the full trust-guided variational network.
    """

    def __init__(self, model: nn.Module, asc_model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of TGVN.
            asc_model: Module for "trust guidance" component of TGVN
        """
        super().__init__()

        self.model = model
        self.asc_model = asc_model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.asc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_image: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        second_kspace: torch.Tensor,
        sens_maps: torch.Tensor,
        delta: torch.Tensor,
        sens_maps_second: torch.Tensor = None,
        num_iter: int = 10,
    ) -> torch.Tensor:

        current_kspace = sens_expand(current_image, sens_maps)
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero)
        soft_dc *= torch.abs(self.dc_weight)
        soft_dc = sens_reduce(soft_dc, sens_maps)
        model_term = self.model(current_image)

        if sens_maps_second is not None:
            b = current_image - self.asc_model(
                sens_reduce(
                    second_kspace,
                    sens_maps_second
                )
            )
        else:
            b = current_image - self.asc_model(
                sens_reduce(
                    second_kspace,
                    sens_maps
                )
            )

        # Conjugate gradient to solve Cx=b
        # where b is given above
        mask_delta = 1 + mask / delta ** 2
        mask_delta_inv = 1 / mask_delta
        x = sens_reduce(
            mask_delta_inv * sens_expand(
                b, sens_maps
            ), sens_maps
        )
        r = b - sens_reduce(
            mask_delta * sens_expand(
                x, sens_maps
            ), sens_maps
        )
        p = r.clone()
        rs_old = inner(r, r)

        for _ in range(num_iter):
            Ap = sens_reduce(
                mask_delta * sens_expand(
                    p, sens_maps
                ), sens_maps
            )
            alpha = rs_old / inner(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rs_new = inner(r, r)
            p = r + rs_new / rs_old * p
            rs_old = rs_new

        soft_asc = torch.abs(self.asc_weight) * x
        return current_image - soft_dc - soft_asc - model_term


class VarNetImage(nn.Module):
    """
    Adapted from the fastMRI implementation.
    fastMRI is copyright-protected by Facebook, Inc. and its affiliates.

    A full E2E-VarNet model implemented in image domain.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
        )

        self.cascades = nn.ModuleList(
            [
                VarNetBlockImage(
                    NormUnet(chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        return_mag: bool = True,
    ) -> torch.Tensor:

        sens_maps = self.sens_net(masked_kspace)
        image_pred = sens_reduce(masked_kspace, sens_maps)

        for cascade in self.cascades:
            image_pred = cascade(
                image_pred,
                masked_kspace,
                mask,
                sens_maps
            )

        # return magnitude or complex-valued image
        if return_mag:
            return complex_abs(image_pred)
        else:
            return torch.view_as_complex(image_pred)


class TGVN_1S(nn.Module):
    """
    A full TGVN model.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for TGVN.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.delta = nn.Parameter(0.1 * torch.ones(1))
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
        )
        self.cascades = nn.ModuleList(
            [
                TGVN_Block(
                    NormUnet(chans, pools), NormUnet(chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        second_kspace: torch.Tensor,
        return_mag: bool = True,
    ) -> torch.Tensor:

        sens_maps = self.sens_net(masked_kspace)
        image_pred = sens_reduce(masked_kspace, sens_maps)

        for cascade in self.cascades:
            image_pred = cascade(
                image_pred,
                masked_kspace,
                mask,
                second_kspace,
                sens_maps,
                self.delta,
            )

        # return magnitude or complex-valued image
        if return_mag:
            return complex_abs(image_pred)
        else:
            return torch.view_as_complex(image_pred)


class TGVN_2S(nn.Module):
    """
    A full TGVN model with 2 sensitivity map estimations.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for TGVN.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()
        self.delta = nn.Parameter(0.1 * torch.ones(1))
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
        )
        self.cascades = nn.ModuleList(
            [
                TGVN_Block(
                    NormUnet(chans, pools), NormUnet(chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        second_kspace: torch.Tensor,
        return_mag: bool = True,
    ) -> torch.Tensor:

        sens_maps = self.sens_net(masked_kspace)
        sens_maps_second = self.sens_net(second_kspace)
        image_pred = sens_reduce(masked_kspace, sens_maps)

        for cascade in self.cascades:
            image_pred = cascade(
                image_pred,
                masked_kspace,
                mask,
                second_kspace,
                sens_maps,
                self.delta,
                sens_maps_second,
            )

        # return magnitude or complex-valued image
        if return_mag:
            return complex_abs(image_pred)
        else:
            return torch.view_as_complex(image_pred)
