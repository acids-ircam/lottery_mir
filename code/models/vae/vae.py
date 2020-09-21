import torch
import numpy as np
import torch.nn as nn
import torch.distributions as distrib
from models.vae.ae import AE

class VAE(AE):
    
    def __init__(self, args):
        super(VAE, self).__init__(args)
        self.encoder = args.encoder
        self.decoder = args.decoder
        self.latent_dims = args.latent_dims
        self.encoder_dims = args.encoder_dims
        self.input_dims = args.input_size
        # Latent gaussians
        #self.mu = nn.Linear(self.encoder_dims, self.latent_dims)
        #self.log_var = nn.Sequential(nn.Linear(self.encoder_dims, self.latent_dims))
        self.gaussian_dec = False
        self.map_latent = None
        # Gaussian decoder
        if (self.gaussian_dec):
            in_prod = np.prod(self.input_dims)
            self.mu_dec = nn.Linear(in_prod, in_prod)
            self.log_var_dec = nn.Sequential(
                nn.Linear(in_prod, in_prod))
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var
    
    def decode(self, z):
        if (self.gaussian_dec):
            n_batch = z.size(0)
            x_vals = self.decoder(z)
            x_vals = x_vals.view(-1, np.prod(self.input_dims))
            mu = self.mu_dec(x_vals)
            log_var = self.log_var_dec(x_vals)
            q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
            eps = q.sample((n_batch, )).detach().to(z.device)
            x_tilde = (log_var.exp().sqrt() * eps) + mu
            x_tilde = x_tilde.view(-1, * self.input_dims)
        else:
            x_tilde = self.decoder(z)
        return x_tilde

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        #z_q_mean, z_q_logvar = z_params
        # Obtain latent samples and latent loss
        z_tilde, z_loss = self.latent(x, z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, z_tilde, z_loss #, z_q_mean, z_q_logvar
    
    def latent(self, x, z_params):
        n_batch = x.size(0)
        # Retrieve mean and var
        mu, log_var = z_params
        # Re-parametrize
        #q = distrib.Normal(torch.zeros(mu.shape[1]), torch.ones(log_var.shape[1]))
        #eps = q.sample((n_batch, )).detach().to(x.device)
        eps = torch.randn_like(mu).detach().to(x.device)
        z = (log_var.exp().sqrt() * eps) + mu
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div / torch.prod(torch.tensor(x.shape[1:]))
        return z, kl_div
