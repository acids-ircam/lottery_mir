import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def sampleTransform(x, transform="softmax"):
    if transform == "softmax":
        return torch.softmax(x, 2)
    elif transform == "doublesoftmax":
        return torch.softmax(torch.softmax(x, 2), 2)
    elif transform == "argmax":
        y = torch.zeros_like(x)
        idx = torch.argmax(x, 2)
        y[:, :, idx] = 1
        return y


class CircularTensor:
    """
    Define a Tensor mapped on a cylinder in order to speed up roll time during
    fast generation loop.
    """
    def __init__(self, tensor, dim):
        self.dim = dim
        self.roll = 0
        self.tensor = tensor
        self.mod = self.tensor.shape[self.dim]

    def __getitem__(self, index):
        index = list(index)
        index[self.dim] = (index[self.dim] + self.roll) % self.mod
        return self.tensor[tuple(index)]

    def __setitem__(self, index, value):
        index = list(index)
        index[self.dim] = (index[self.dim] + self.roll) % self.mod
        self.tensor[tuple(index)] = value

    def rollBy(self, amount):
        self.roll -= amount
        self.roll = self.roll % self.mod

    def __str__(self):
        return str(self.tensor)


class Conv1d(nn.Conv1d):
    """
    Modification of Conv1d to support linearization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linearized_weight = None

    def linearize_weight(self):
        self._linearized_weight = self.weight.transpose(
            1, 2).contiguous().view(self.out_channels, -1)

    def incremental_forward(self, input):
        if self._linearized_weight is None:
            self.linearize_weight()
        bsz = input.shape[0]
        out = nn.functional.linear(input.view(bsz, -1),
                                   self._linearized_weight, self.bias)
        return out.view(bsz, 1, -1)


class ResidualBlock(nn.Module):
    """
    Implementation of a Residual Block as described in the deep voice 3 variant
    of the Wavenet vocoder.

    Parameters
    ----------

    dilation: int
        Defines the dilation factor of the dilated convolution
    """
    def __init__(self,
                 dilation,
                 res_size,
                 skp_size,
                 cdt_size,
                 last_conv=False):
        super().__init__()

        self.last_conv = last_conv
        self.dilated_conv = Conv1d(res_size,
                                   2 * res_size,
                                   2,
                                   padding=dilation,
                                   dilation=dilation)

        self.cdt_conv = Conv1d(cdt_size, 2 * res_size, 1)

        if not last_conv:
            self.res_conv = Conv1d(res_size, res_size, 1)
        self.skp_conv = Conv1d(res_size, skp_size, 1)

    def forward(self, x, c, incremental=False):
        """
        Forward pass of the residual block.

        Parameters
        ----------

        x: Tensor
            Either residual from previous residual block or output of the
            very first Convolution layer of the network. Must be of shape
            [B x hp.RES_SIZE x T]

        c: Tensor
            Local condition. Must be of shape [B x cond_size x T]


        Returns
        -------

        residual: Tensor
            Output of the residual convolution. Shape [B x hp.RES_SIZE x T]

        skip: Tensor
            Output of the skip convolution. Shape [B x hp.SKP_SIZE x T]
        """
        residual = x.clone()

        if incremental:
            residual = residual[:, -1:, :]
            x = self.dilated_conv.incremental_forward(x)
            xa, xb = x.split(residual.shape[2], 2)

            c = self.cdt_conv.incremental_forward(c)
            ca, cb = c.split(residual.shape[2], 2)

        else:
            x = self.dilated_conv(x)[:, :, :residual.shape[-1]]
            xa, xb = x.split(residual.shape[1], 1)

            c = self.cdt_conv(c)
            ca, cb = c.split(residual.shape[1], 1)



        gate_out = torch.tanh(xa + ca) \
                 * torch.sigmoid(xb + cb)

        if incremental:
            skip = self.skp_conv.incremental_forward(gate_out)
            if not self.last_conv:
                residual += self.res_conv.incremental_forward(gate_out)
        else:
            skip = self.skp_conv(gate_out)
            if not self.last_conv:
                residual += self.res_conv(gate_out)

        return residual, skip


class Wavenet(nn.Module):
    """
    Implementation of the deep voice 3 variant of the wavenet model.

    Parameters:
    -----------

    cycle_nb: int
        Number of dilation resets

    n_layer: int
        Number of ... layers

    in_size: int
        Number of channels of input waveform (quantized)
    
    res_size: int
        Dimensionality of residual connection

    skp_size: int
        Dimensionality of skip connection

    cdt_size: int
        Dimensionality of condition

    out_size: int
        Usually the same as input size, but who knows
    
    dim_reduction: int
        Hop length if its conditionned on spectrums, else total network stride

    """
    def __init__(self, cycle_nb, n_layer, in_size, res_size, skp_size,
                 cdt_size, out_size, dim_reduction):
        super().__init__()

        self.receptive_field = cycle_nb * (2**(n_layer // cycle_nb + 1))
        self.first_conv = Conv1d(in_size, res_size, 2, padding=1)

        rs = []
        for i in range(n_layer):
            rs.append(
                ResidualBlock(
                    2**(i % (n_layer // cycle_nb)),
                    res_size,
                    skp_size,
                    cdt_size,
                    last_conv=i == n_layer - 1,
                ))

        self.ResidualStack = nn.ModuleList(rs)

        self.last_conv = nn.Sequential(
            nn.ReLU(), Conv1d(skp_size, skp_size, 1, bias=False), nn.ReLU(),
            Conv1d(skp_size, out_size, 1, bias=False))

        self.last_conv[-1].unprunable = True

        self.cycle_nb = cycle_nb
        self.n_layer = n_layer
        self.in_size = in_size
        self.res_size = res_size
        self.skp_size = skp_size
        self.cdt_size = cdt_size
        self.out_size = out_size
        self.dim_reduction = dim_reduction

        skipped = 0
        for p in self.parameters():
            try:
                nn.init.kaiming_normal_(p)
            except:
                skipped += 1
        print("Skipped %d parameters during initialisation" % (skipped))

    def forward(self, x, c=None, expanded=False, incremental=False):
        """
        Forward pass of the wavenet model.

        Parameters
        ----------

        x: Tensor
            Audio sample to reconstruct. Must be of shape [B x hp.decoder_hparams.in_size x T]

        c: Tensor
            Local condition. Must be of shape [B x cond_size x T]


        expanded: bool
            Defines wheither the local conditionning tensor is already expanded to
            the waveform shape or not, as it changes wheither we are in eval or
            training mode.
        """

        if c is None:
            c = torch.randn(
                x.shape[0],
                self.cdt_size,
                x.shape[-1] // self.dim_reduction,
            ).to(x)

        x = torch.tanh(self.first_conv(x)[:, :, :-1])

        if not expanded:
            stride = self.dim_reduction
            c = c.repeat_interleave(stride).view(c.shape[0], c.shape[1],
                                                 c.shape[2] * stride)

        res = x
        skip = None

        for layer in self.ResidualStack:
            res, new_skip = layer(res, c)
            if skip is not None:
                skip += new_skip
            else:
                skip = new_skip

        skip = self.last_conv(skip)
        return skip

    def generate_fast(self, c, expanded=False, transform="softmax"):
        """
        Fast Wavenet generation algorithm, using cached and linearized convolutions.
        The batch generation is available and has been tested.

        Parameters
        ----------

        c: Tensor
            Local condition. Must be of shape [B x cond_size x T]

        Returns
        -------

        y: Tensor
            Predicted sampled. Shape [B x hp.decoder_hparams.in_size x T]
        """

        bs = c.shape[0]
        idx = torch.arange(bs)

        if not expanded:
            stride = self.dim_reduction
            c = c.repeat_interleave(stride).reshape(
                c.shape[0], c.shape[1], c.shape[2] * stride).contiguous()

        c = c.transpose(1, 2).contiguous()

        T = c.shape[1]

        output = torch.zeros(c.shape[0], T, self.in_size).to(c.device)

        cache = []

        for layer in self.ResidualStack:
            tens = torch.zeros(c.shape[0], layer.dilated_conv.dilation[0],
                               self.res_size).to(c.device)
            cache.append(CircularTensor(tens, dim=1))

        relu = nn.ReLU()

        with torch.no_grad():
            for t in tqdm(range(1, T - 1), desc="generating"):
                res = output[:, t - 1:t + 1, :]
                res = torch.tanh(self.first_conv.incremental_forward(res))

                c_cropped = c[:, t:t + 1, :]

                skip = None

                for i, layer in enumerate(self.ResidualStack):
                    old_res = res.clone()

                    res, new_skip = layer(torch.cat(
                        [cache[i][:, 0, :].unsqueeze(1), res], 1),
                                          c_cropped,
                                          incremental=True)

                    if skip is not None:
                        skip += new_skip
                    else:
                        skip = new_skip

                    cache[i].rollBy(-1)
                    cache[i][:, -1, :] = old_res[:, 0, :]

                skip = self.last_conv[1].incremental_forward(relu(skip))
                pred = self.last_conv[-1].incremental_forward(relu(skip))

                sample = torch.distributions.categorical.Categorical(
                    sampleTransform(pred[:, -1:, :],
                                    transform).view(bs, -1)).sample()
                output[idx, t + 1, sample] = 1
        output = torch.argmax(output, -1).float()
        output /= self.in_size
        output *= 2
        output -= 1
        return output
