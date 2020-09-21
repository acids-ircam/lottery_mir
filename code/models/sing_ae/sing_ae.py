import torch
from torch import nn

from models.sing_ae.utils import WindowedConv1d, WindowedConvTranpose1d

class ConvolutionalDecoder(nn.Module):
    """
    Convolutional decoder that takes a downsampled embedding and turns it
    into a waveform.
    Together with :class:`ConvolutionalEncoder`, it forms a
    :class:`ConvolutionalAE`

    Arguments:
        channels (int): number of channels accross all the inner layers
        stride (int): stride of the final :class:`nn.ConvTranspose1d`
        dimension (int): dimension of the embedding
        kernel_size (int): size of the kernel of the final
            :class:`nn.ConvTranspose1d`
        context_size (int): kernel size of the first convolution,
            this is called a context as one can see it as providing
            information about the previous and following embeddings
        rewrite_layers (int): after the first convolution, perform
            `rewrite_layers` `1x1` convolutions
        window_name (str or None): name of the window used to smooth
            the convolutions. See :func:`sing.dsp.get_window`
        squared_window (bool): if `True`, square the smoothing window
    """

    def __init__(self,
                 channels=4096,
                 stride=256,
                 dimension=128,
                 kernel_size=1024,
                 context_size=9,
                 rewrite_layers=2,
                 window_name="hann",
                 squared_window=True):
        super(ConvolutionalDecoder, self).__init__()
        layers = []
        layers.extend([
            nn.Conv1d(
                in_channels=dimension,
                out_channels=channels,
                kernel_size=context_size),
            nn.ReLU()
        ])
        for rewrite in range(rewrite_layers):
            layers.extend([
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=1),
                nn.ReLU()
            ])

        conv_tr = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size - stride)
        if window_name is not None:
            conv_tr = WindowedConvTranpose1d(conv_tr, window_name,
                                             squared_window)
        layers.append(conv_tr)
        self.layers = nn.Sequential(*layers)
        self.layers[-1].unprunable = True
        self.context_size = context_size
        self.stride = stride
        self.kernel_size = kernel_size
        self.strip = kernel_size - stride + (context_size - 1) * stride // 2

    def __repr__(self):
        return "ConvolutionalDecoder({})".format(repr(self.layers))

    def forward(self, embeddings):
        return self.layers.forward(embeddings)

    def wav_length(self, embedding_length):
        """
        Given an embedding of a certain size `embedding_length`,
        returns the length of the wav that would be generated from it.
        """
        return (embedding_length - self.context_size + 2
                ) * self.stride - self.kernel_size

    def embedding_length(self, wav_length):
        """
        Return the embedding length necessary to generate a wav of length
        `wav_length`.
        """
        return self.context_size - 2 + (
            wav_length + self.kernel_size) // self.stride


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional encoder that takes a waveform and turns it
    into a downsampled embedding.
    Together with :class:`ConvolutionalDecoder`, it forms a
    :class:`ConvolutionalAE`

    Arguments:
        channels (int): number of channels accross all the inner layers
        stride (int): stride of the initial :class:`nn.Conv1d`
        dimension (int): dimension of the embedding
        kernel_size (int): size of the kernel of the initial
            :class:`nn.Conv1d`
        rewrite_layers (int): after the first convolution, perform
            `rewrite_layers` `1x1` convolutions.
        window_name (str or None): name of the window used to smooth
            the convolutions. See :func:`sing.dsp.get_window`
        squared_window (bool): if `True`, square the smoothing window
    """

    def __init__(self,
                 channels=4096,
                 stride=256,
                 dimension=128,
                 kernel_size=1024,
                 rewrite_layers=2,
                 window_name="hann",
                 squared_window=True):
        super(ConvolutionalEncoder, self).__init__()
        layers = []
        conv = nn.Conv1d(
            in_channels=1,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride)
        if window_name is not None:
            conv = WindowedConv1d(conv, window_name, squared_window)
        layers.extend([conv, nn.ReLU()])
        for rewrite in range(rewrite_layers):
            layers.extend([
                nn.Conv1d(
                    in_channels=channels, out_channels=channels,
                    kernel_size=1),
                nn.ReLU()
            ])

        layers.append(
            nn.Conv1d(
                in_channels=channels, out_channels=dimension, kernel_size=1))
        self.layers = nn.Sequential(*layers)
        self.layers[-1].unprunable = True

    def __repr__(self):
        return "ConvolutionalEncoder({!r})".format(self.layers)

    def forward(self, signal):
        return self.layers.forward(signal)


class ConvolutionalAE(nn.Module):
    """
    Convolutional autoencoder made from :class:`ConvolutionalEncoder` and
    :class:`ConvolutionalDecoder`.

    Arguments:
        channels (int): number of channels accross all the inner layers
        stride (int): downsampling stride going from the waveform
            to the embedding
        dimension (int): dimension of the embedding
        kernel_size (int): kernel size of the initial convolution
            and last conv transpose
        context_size (int): kernel size of the first
            convolution of the decoder
        rewrite_layers (int): after the first convolution, perform
            `rewrite_layers` `1x1` convolutions, both in the encoder
            and decoder.
        window_name (str or None): name of the window used to smooth
            the convolutions. See :func:`sing.dsp.get_window`
        squared_window (bool): if `True`, square the smoothing window
    """

    def __init__(self,
                 channels=4096,
                 stride=256,
                 dimension=128,
                 kernel_size=1024,
                 context_size=2,
                 rewrite_layers=5,
                 window_name="hann",
                 squared_window=True):
        super(ConvolutionalAE, self).__init__()
        self.encoder = ConvolutionalEncoder(
            channels=channels,
            stride=stride,
            dimension=dimension,
            kernel_size=kernel_size,
            rewrite_layers=rewrite_layers,
            window_name=window_name,
            squared_window=squared_window)
        self.decoder = ConvolutionalDecoder(
            channels=channels,
            stride=stride,
            dimension=dimension,
            kernel_size=kernel_size,
            context_size=context_size,
            rewrite_layers=rewrite_layers,
            window_name=window_name,
            squared_window=squared_window)

#        print(self)

    def encode(self, signal):
        """
        Returns the embedding for the waveform `signal`.
        """
        return self.encoder.forward(signal)

    def decode(self, embeddings):
        """
        Return the waveforms from `embeddings`
        """
        return self.decoder.forward(embeddings).squeeze(1)

    def forward(self, signal):
        z = self.encode(signal)
        return self.decode(z), z

    def __repr__(self):
        return "ConvolutionalAE(encoder={!r},decoder={!r})".format(
            self.encoder, self.decoder)

class ConvolutionalVAE(ConvolutionalAE):

    def __init__(self,
                 channels,
                 stride,
                 dimension,
                 kernel_size,
                 context_size,
                 rewrite_layers,
                 window_name,
                 squared_window,
                 latent_dims,
                 time_dims):
        super().__init__(channels,
                 stride,
                 dimension,
                 kernel_size,
                 context_size,
                 rewrite_layers,
                 window_name,
                 squared_window)
        self.encoder = ConvolutionalEncoder(
            channels=channels,
            stride=stride,
            dimension=dimension,
            kernel_size=kernel_size,
            rewrite_layers=rewrite_layers,
            window_name=window_name,
            squared_window=squared_window)
#        self.decoder = ConvolutionalDecoder(
#            channels=channels,
#            stride=stride,
#            dimension=dimension,
#            kernel_size=kernel_size,
#            context_size=context_size,
#            rewrite_layers=rewrite_layers,
#            window_name=window_name,
#            squared_window=squared_window)
        self.decoder = ConvolutionalDecoder(
            channels=channels,
            stride=stride,
            dimension=latent_dims,
            kernel_size=kernel_size,
            context_size=context_size,
            rewrite_layers=rewrite_layers,
            window_name=window_name,
            squared_window=squared_window)
        self.dimension = dimension
        self.latent_dims = latent_dims
#        self.mu = nn.Linear(dimension*time_dims, latent_dims*time_dims)
        self.mu = nn.Conv1d(dimension,latent_dims,1,1)
#        self.log_var = nn.Linear(dimension*time_dims, latent_dims*time_dims)
        self.log_var = nn.Conv1d(dimension,latent_dims,1,1)

#        self.unflat = nn.Linear(latent_dims*time_dims, dimension*time_dims)
#        self.unflat = nn.Linear(latent_dims*time_dims, dimension*time_dims)
        #Gaussian decoder
#        self.mu_dec = nn.Conv1d(1, 1, kernel_size = 1)
#        self.log_var_dec = nn.Conv1d(1, 1, kernel_size = 1)
        
    def encode(self, signal):
        x = self.encoder(signal)

        return self.mu(x), self.log_var(x)

    def decode(self, z):
        #unflatten
#        z = self.unflat(z)
        n_batch = z.shape[0]
        z = z.view(n_batch,self.latent_dims, -1)

        return self.decoder(z).squeeze(1)
    
    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        z_tilde, z_loss = self.latent(x, z_params)
        x_tilde = self.decode(z_tilde)

        return x_tilde, z_tilde, z_loss
    
    
    def latent(self, x, z_params):
        n_batch = x.shape[0]
        # Retrieve mean and var
        mu, log_var = z_params
        # Re-parametrize
        eps = torch.randn_like(mu).detach().to(x.device)
        z = (log_var.exp().sqrt() * eps) + mu
        #print(z.shape)
        # Compute KL divergence
        kl_div = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))/265
        kl_div = kl_div / n_batch
        return z, kl_div
