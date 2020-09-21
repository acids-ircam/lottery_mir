import torch
import torch.nn as nn

class RegressionAE(nn.Module):
    """
    Definition of AE model for all regression tasks.
    """
    
    def __init__(self, ae_model, latent_dims, regression_dims, recons_loss, regressor = None, regressor_name = '', **kwargs):
        super(RegressionAE, self).__init__(**kwargs)
        self.ae_model = ae_model
        self.recons_loss = recons_loss
        self.latent_dims = latent_dims
        self.regression_dims = regression_dims
        if (regressor is None):
            self.regression_model = nn.Sequential(
                    nn.Linear(latent_dims, latent_dims * 2),
                    nn.ReLU(), nn.BatchNorm1d(latent_dims * 2),
                    nn.Linear(latent_dims * 2, latent_dims * 2),
                    nn.ReLU(), nn.BatchNorm1d(latent_dims * 2),
                    nn.Linear(latent_dims * 2, regression_dims),
                    nn.ReLU(),
                    nn.Hardtanh(min_val=0, max_val=1.))
            regressor_name = 'mlp'
        else:
            self.regression_model = regressor
        self.regressor = regressor_name
    
    def forward(self, x):
        # Auto-encode
        x_tilde, z_tilde, z_loss = self.ae_model(x)
        # Perform regression on params
        p_tilde = self.regression_model(z_tilde)
        return p_tilde
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, _, _) in loader:
            # Send to device
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            # Auto-encode
            x_tilde, z_tilde, z_loss = self.ae_model(x)
            # Reconstruction loss
            rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            if (self.regressor == 'mlp'):
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
            # Final loss
            b_loss = (rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss)).mean(dim=0)
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss_params, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, y, _, _) in loader:
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                # Auto-encode
                x_tilde, z_tilde, z_loss = self.ae_model(x)
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
                full_loss += reg_loss
            full_loss /= len(loader)
        return full_loss

class DisentanglingAE(RegressionAE):
    """
    Definition of regression AE model with the added 
    """
    
    def __init__(self, ae_model, latent_dims, regression_dims, recons_loss, regressor = None, regressor_name = '', disentangling = None, semantic_dim = -1, **kwargs):
        super(DisentanglingAE, self).__init__(ae_model, latent_dims, regression_dims, recons_loss, regressor, regressor_name)
        # Disentangling model
        self.disentangling = disentangling
        # Semantic dim to evaluate
        self.semantic_dim = semantic_dim
    
    def forward(self, x):
        # Auto-encode
        #x_tilde, z_tilde, z_loss = self.ae_model(x)
        # Disentangling part
        #z_tilde, _ = self.disentangling(z_tilde)
        # Encode the inputs
        z_params = self.ae_model.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, z_loss = self.ae_model.latent(x, z_params)
        # Perform disentangling
        z_tilde, _ = self.disentangling(z_tilde)
        # Decode the samples
        #x_tilde = self.decode(z_tilde)
        # Perform regression on params
        p_tilde = self.regression_model(z_tilde)
        return p_tilde
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, meta, _) in loader:
            # Send to device
            x, y, meta = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), meta.to(args.device, non_blocking=True)
            # Extract current meta-tag
            meta = meta[:, self.semantic_dim].squeeze(1)
            target = meta[:, 1].long()
            # Separate examples
            loss_mask = 1 - meta[:, 2]
            observed = loss_mask.eq(1)
            unknown = loss_mask.eq(0)
            # Auto-encode
            #x_tilde, z_tilde, z_loss = self.ae_model(x)
            # Reconstruction loss
            #rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            # Disentangling part
            #z_tilde, dis_loss = self.disentangling(z_tilde, (meta, target, observed, unknown))            
            # Encode the inputs
            z_params = self.ae_model.encode(x)
            # Obtain latent samples and latent loss
            z_tilde, z_loss = self.ae_model.latent(x, z_params)
            # Disentangling part
            z_tilde, dis_loss = self.disentangling(z_tilde, (meta, target, observed, unknown))            
            # Decode the samples
            x_tilde = self.ae_model.decode(z_tilde)
            # Reconstruction loss
            rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            # Regression part
            if (self.regressor == 'mlp'):
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
            # Final loss
            b_loss = (rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss) + (args.beta * dis_loss)).mean(dim=0)
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss_params, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, y, meta, _) in loader:
                # Send to device
                x, y, meta = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), meta.to(args.device, non_blocking=True)
                # Extract current meta-tag
                #meta = meta[:, self.semantic_dim].squeeze(1)
                #target = meta[:, 1].long()
                # Separate examples
                #loss_mask = 1 - meta[:, 2]
                #observed_examples = loss_mask.eq(1)
                #unknown_examples = loss_mask.eq(0)
                # Auto-encode
                #x_tilde, z_tilde, z_loss = self.ae_model(x)
                # Disentangling part
                #z_tilde, _ = self.disentangling(z_tilde)
                # Encode the inputs
                z_params = self.ae_model.encode(x)
                # Obtain latent samples and latent loss
                z_tilde, z_loss = self.ae_model.latent(x, z_params)
                # Disentangling part
                z_tilde, _ = self.disentangling(z_tilde)            
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Regression loss
                reg_loss = loss_params(p_tilde, y)
                #e Compute full loss
                full_loss += reg_loss
            full_loss /= len(loader)
        return full_loss
    
class AE(nn.Module):
    
    def __init__(self, args):
        super(AE, self).__init__()
        self.encoder = args.encoder
        #self.map_latent = nn.Linear(args.encoder_dims, args.latent_dims)
        self.decoder = args.decoder
        self.latent_dims = args.latent_dims
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, z):
        return self.decoder(z)
    
    def regularize(self, z):
        z = self.map_latent(z)
        return z, torch.zeros(z.shape[0]).to(z.device).mean()

    def forward(self, x):
        # Encode the inputs
        z = self.encode(x)
        # Potential regularization
        z_tilde, z_loss = self.regularize(z)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, z_tilde, z_loss
