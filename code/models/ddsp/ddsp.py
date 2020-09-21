import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


def sanity_check(f0, lo):
    assert len(
        f0.shape
    ) == 3, "f0 input must be 3-dimensional, but is %d-dimensional." % (len(
        f0.shape))
    assert len(
        lo.shape
    ) == 3, "lo input must be 3-dimensional, but is %d-dimensional." % (len(
        lo.shape))


class Reverb(nn.Module):
    """
    Model of a room impulse response. Two parameters controls the shape of the
    impulse reponse: wet/dry amount and the exponential decay.

    Parameters
    ----------

    size: int
        Size of the impulse response (samples)
    """
    def __init__(self, size):
        super().__init__()
        self.size = size

        self.register_buffer("impulse", torch.rand(1, size) * 2 - 1)
        self.register_buffer("identity", torch.zeros(1, size))

        self.impulse[:, 0] = 0
        self.identity[:, 0] = 1

        self.decay = nn.Parameter(torch.Tensor([2]), requires_grad=True)
        self.wetdry = nn.Parameter(torch.Tensor([4]), requires_grad=True)

    def forward(self, impulse):
        idx = torch.sigmoid(self.wetdry) * self.identity
        imp = torch.sigmoid(1 - self.wetdry) * impulse
        dcy = torch.exp(-(torch.exp(self.decay)+2) * \
              torch.linspace(0,1, self.size).to(self.decay.device))

        return idx + imp * dcy


def mod_sigmoid(x):
    """
    Implementation of the modified sigmoid described in the original article.

    Parameters
    ----------

    x: Tensor
        Input tensor, of any shape

    Returns
    -------

    Tensor:
        Output tensor, shape of x
    """
    return 2 * torch.sigmoid(x)**np.log(10) + 1e-7


class MLP(nn.Module):
    """
    Implementation of a Multi Layer Perceptron, as described in the
    original article (see README)

    Parameters
    ----------

    in_size: int
        Input size of the MLP
    out_size: int
        Output size of the MLP
    loop: int
        Number of repetition of Linear-Norm-ReLU
    """
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        model = []

        model.extend([
            nn.Linear(in_size, out_size),
            nn.modules.normalization.LayerNorm(out_size),
            nn.ReLU()
        ])

        for i in range(loop - 1):
            model.extend([
                nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ])

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        kernel,
        dilation,
        stride,
    ):
        super().__init__()

        convs = []
        n_conv = len(stride)
        channels = [in_dim] + (n_conv - 1) * [hidden_dim] + [out_dim * 2]

        if isinstance(stride, int):
            stride = [stride] * n_conv

        if isinstance(dilation, int):
            dilation = [dilation] * n_conv

        for i in range(n_conv):
            convs.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel,
                    stride=stride[i],
                    dilation=dilation[i],
                    padding=((kernel - 1) * dilation[i]) // 2,
                ))
            if i != n_conv - 1:
                convs.append(nn.ReLU())
                convs.append(nn.BatchNorm1d(channels[i + 1]))

        self.net = nn.Sequential(*convs)
        self.net[-1].unprunable = True

    def forward(self, x):
        out = self.net(x)
        mean, logvar = torch.split(out, out.shape[1] // 2, 1)
        z = torch.randn_like(mean) * torch.exp(logvar) + mean
        kl_loss = mean**2 + torch.exp(logvar) - logvar - 1
        return z, kl_loss.mean()


class Decoder(nn.Module):
    """
    Decoder of the architecture described in the original architecture.

    Parameters
    ----------

    hidden_size: int
        Size of vectors inside every MLP + GRU + Dense
    n_partial: int
        Number of partial involved in the harmonic generation. (>1)
    filter_size: int
        Size of the filter used to shape noise.


    """
    def __init__(self, hidden_size, n_partial, filter_size, latent_size):
        super().__init__()
        self.f0_MLP = MLP(1, hidden_size)
        self.f0_MLP.model[-3].unprunable = True
        self.lo_MLP = MLP(1, hidden_size)
        self.lo_MLP.model[-3].unprunable = True

        self.gru = nn.GRU(2 * hidden_size + latent_size,
                          hidden_size,
                          batch_first=True)

        self.fi_MLP = MLP(hidden_size, hidden_size)
        self.fi_MLP.model[-3].unprunable = True

        self.dense_amp = nn.Linear(hidden_size, 1)
        self.dense_alpha = nn.Linear(hidden_size, n_partial)
        self.dense_filter = nn.Linear(hidden_size, filter_size // 2 + 1)
        # Put to unprunable
        self.dense_amp.unprunable = True
        self.dense_alpha.unprunable = True
        self.dense_filter.unprunable = True

        self.n_partial = n_partial

    def forward(self, f0, lo, z):
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)

        z = z.permute(0, 2, 1)

        cat_inputs = torch.cat([f0, lo, z], -1)
        x = self.gru(cat_inputs)

        if isinstance(x, tuple):
            x = x[0]

        x = self.fi_MLP(x)

        amp = mod_sigmoid(self.dense_amp(x))
        alpha = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))

        alpha = alpha / torch.sum(alpha, -1).unsqueeze(-1)

        return amp, alpha, filter_coeff


class NeuralSynth(nn.Module):
    """
    Implementation of a parametric Harmonic + Noise + IR reverb synthesizer,
    whose parameters are controlled by the previously implemented decoder.
    """
    def __init__(
        self,
        latent_hidden_dim: int,
        latent_size: int,
        kernel_size: int,
        dilation: list,
        stride: list,
        hidden_size: int,
        n_partial: int,
        filter_size: int,
        block_size: int,
        samplerate: int,
        sequence_size: int,
        fft_scales: list,
    ):

        super().__init__()

        self.latent_hidden_dim = latent_hidden_dim
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.hidden_size = hidden_size
        self.n_partial = n_partial
        self.filter_size = filter_size
        self.block_size = block_size
        self.samplerate = samplerate
        self.sequence_size = sequence_size
        self.fft_scales = fft_scales

        self.encoder = Encoder(
            in_dim=1,
            hidden_dim=latent_hidden_dim,
            out_dim=latent_size,
            kernel=kernel_size,
            dilation=dilation,
            stride=stride,
        )

        self.decoder = Decoder(
            hidden_size=hidden_size,
            n_partial=n_partial,
            filter_size=filter_size,
            latent_size=latent_size,
        )

        self.condition_upsample = nn.Upsample(scale_factor=block_size,
                                              mode="linear")

        for n, p in self.named_parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                # print(f"Skipped initialization of {n}")
                pass

        self.register_buffer(
            "k",
            torch.arange(1, n_partial + 1).reshape(1, 1, -1).float(),
        )

        self.windows = nn.ParameterList(
        nn.Parameter(torch.from_numpy(
                np.hanning(scale)
            ).float(), requires_grad=False)\
            for scale in fft_scales)

        self.register_buffer(
            "filter_window",
            torch.hann_window(filter_size).roll(filter_size // 2, -1),
        )

        self.impulse = Reverb(sequence_size * block_size)

    def forward(
        self,
        x,
        cdt=None,
        noise_pass=True,
        conv_pass=True,
        phi=None,
        only_y=True,
    ):
        self.decoder.gru.flatten_parameters()

        if cdt is not None:
            f0, lo = torch.split(cdt, 1, -1)
        else:
            # DUMMY F0 AND LOUDNESS COMPUTING
            lo = x.reshape(x.shape[0], -1, 160).pow(2).mean(-1, keepdim=True)
            f0 = lo.clone()

        bs = f0.shape[0]

        x = x.reshape(bs, 1, -1)

        sanity_check(f0, lo)

        # CREATE LATENT REPRESENTATION ########################################
        z, kl_loss = self.encoder(x)
        # CREATE HARMONIC PART ################################################
        amp, alpha, filter_coef = self.decoder(f0, lo, z)

        f0 = self.condition_upsample(f0.transpose(1, 2))
        f0 = f0.squeeze(1) / self.samplerate

        amp = self.condition_upsample(amp.transpose(1, 2)).squeeze(1)
        alpha = self.condition_upsample(alpha.transpose(1, 2)).transpose(1, 2)

        if phi is None:
            phi = torch.zeros(f0.shape).to(f0.device)

            for i in np.arange(1, phi.shape[-1]):
                phi[:, i] = 2 * np.pi * f0[:, i] + phi[:, i - 1]

        phi = phi.unsqueeze(-1).expand(alpha.shape)

        antia_alias = (self.k * f0.unsqueeze(-1) < .5).float()

        y = amp * torch.sum(antia_alias * alpha * torch.sin(self.k * phi), -1)

        # FREQUENCY SAMPLING FILTERING #########################################
        noise = torch.from_numpy(np.random.uniform(-1,1,y.shape))\
                     .float().to(y.device)/1000

        noise = noise.reshape(-1, self.filter_size)

        S_noise = torch.rfft(noise, 1)
        S_noise = S_noise.reshape(bs, -1, self.filter_size // 2 + 1, 2)

        filter_coef = filter_coef.reshape([-1, self.filter_size // 2 + 1, 1])
        filter_coef = filter_coef.expand([-1, self.filter_size // 2 + 1, 2])
        filter_coef = filter_coef.contiguous()
        filter_coef[:, :, 1] = 0

        h = torch.irfft(filter_coef, 1, signal_sizes=(self.filter_size, ))
        h_w = self.filter_window.unsqueeze(0) * h
        H = torch.rfft(h_w, 1).reshape(bs, -1, self.filter_size // 2 + 1, 2)

        S_filtered_noise = torch.zeros_like(H)

        S_filtered_noise[
            ..., 0] = H[..., 0] * S_noise[..., 0] - H[..., 1] * S_noise[..., 1]
        S_filtered_noise[
            ..., 1] = H[..., 0] * S_noise[..., 1] + H[..., 1] * S_noise[..., 0]
        S_filtered_noise = S_filtered_noise.reshape(
            -1,
            self.filter_size // 2 + 1,
            2,
        )

        filtered_noise = torch.irfft(S_filtered_noise, 1)
        filtered_noise = filtered_noise[:, :self.filter_size].reshape(bs, -1)

        y += float(noise_pass) * filtered_noise

        # CONVOLUTION WITH AN IMPULSE RESPONSE #################################
        y = nn.functional.pad(y, (0, self.block_size * self.sequence_size))
        Y_S = torch.rfft(y, 1)

        rand_y = torch.rand(1, self.sequence_size * self.block_size) * 2 - 1
        rand_y = rand_y.to(y.device)

        impulse = self.impulse(rand_y)

        impulse = nn.functional.pad(impulse,
                                    (0, self.block_size * self.sequence_size))

        if y.shape[-1] > self.sequence_size * self.block_size:
            impulse = nn.functional.pad(
                impulse,
                (0, y.shape[-1] - impulse.shape[-1]),
            )

        if conv_pass:
            IR_S = torch.rfft(torch.tanh(impulse), 1).expand_as(Y_S)
        else:
            IR_S = torch.rfft(torch.tanh(impulse.detach()), 1).expand_as(Y_S)

        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[...,
                 0] = Y_S[..., 0] * IR_S[..., 0] - Y_S[..., 1] * IR_S[..., 1]
        Y_S_CONV[...,
                 1] = Y_S[..., 0] * IR_S[..., 1] + Y_S[..., 1] * IR_S[..., 0]

        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1], ))

        y = y[:, :-self.block_size * self.sequence_size]
        S_filtered_noise = S_filtered_noise.reshape(
            bs,
            -1,
            self.filter_size // 2 + 1,
            2,
        )
        if only_y:
            return y
        else:
            return y, amp, alpha, S_filtered_noise, kl_loss

    def multiScaleFFT(self,
                      x,
                      overlap=75 / 100,
                      amp=lambda x: x[:, :, :, 0]**2 + x[:, :, :, 1]**2):
        stfts = []
        for i, scale in enumerate(self.fft_scales):
            stfts.append(
                amp(
                    torch.stft(
                        x,
                        n_fft=scale,
                        window=self.windows[i],
                        hop_length=int((1 - overlap) * scale),
                        center=False,
                    )))
        return stfts


class IncrementalNS(nn.Module):
    def __init__(self, NS):
        super().__init__()
        self.NS = NS

    def forward(self, f0, lo, hx):
        return self.NS.decoder(f0, lo, hx)
