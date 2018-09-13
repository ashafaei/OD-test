import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
"""
    The VAE code is based on
    https://github.com/pytorch/examples/blob/master/vae/main.py
    The q(x|z)-decoder is a bernoulli distribution rather than a Gaussian.
"""

class Generic_AE(nn.Module):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(Generic_AE, self).__init__()

        kernel_size = 3
        all_channels = []
        current_channels = 64
        nonLin = nn.ELU
        self.epoch_factor = max(1, n_hidden/256)
        self.default_sigmoid = False
        max_pool_layers = [i%2==0 for i in range(depth)]
        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]
        in_spatial_size = dims[1]
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, padding=(kernel_size-1)/2))
            modules.append(nn.BatchNorm2d(current_channels))
            modules.append(nonLin())
            in_channels = current_channels
            all_channels.append(current_channels)
            if max_pool_layers[i]:
                modules.append(nn.MaxPool2d(2))
                current_channels = min(current_channels * 2, max_channels)
                remainder_layers.append(in_spatial_size % 2)
                in_spatial_size = math.floor(in_spatial_size/2)

        # Final layer
        modules.append(nn.Conv2d(in_channels, n_hidden, kernel_size=kernel_size, padding=(kernel_size-1)/2))
        modules.append(nn.BatchNorm2d(n_hidden))
        modules.append(nonLin())
        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = in_channels / 2
        current_index = len(all_channels)-1
        r_ind = len(remainder_layers)-1
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, all_channels[current_index], kernel_size=kernel_size, padding=(kernel_size-1)/2))
            modules.append(nn.BatchNorm2d(all_channels[current_index]))
            modules.append(nonLin())
            if max_pool_layers[i]:
                modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
                if remainder_layers[r_ind] > 0:
                    modules.append(nn.ZeroPad2d((1,0,1,0)))
                r_ind -= 1 

            in_channels = all_channels[current_index]
            current_index -= 1
        # Final layer
        modules.append(nn.Conv2d(in_channels, dims[0], kernel_size=kernel_size, padding=(kernel_size-1)/2))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        n_samples = x.size(0)
        code = self.encoder(x)
        out = code.view(n_samples, -1) # flatten to vectors.
        return out

    def forward(self, x, sigmoid=False):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if sigmoid or self.default_sigmoid:
            dec = F.sigmoid(dec)
        return dec

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 240 * self.epoch_factor
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class Generic_VAE(Generic_AE):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256):
        super(Generic_VAE, self).__init__(dims, max_channels, depth, 2*n_hidden)
        self.fc_e_mu  = nn.Linear(2*n_hidden, n_hidden)
        self.fc_e_std = nn.Linear(2*n_hidden, n_hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        n_samples = x.size(0)
        h_out = self.encoder(x)
        code  = self.fc_e_mu(h_out.view(n_samples, -1))
        return code

    def forward(self, x):
        enc     = self.encoder(x)
        n_size  = enc.size(0)
        mu, logvar  = self.fc_e_mu(enc.view(n_size, -1)), self.fc_e_std(enc.view(n_size, -1))
        self.last_mu  = mu
        self.last_std = logvar
        z           = self.reparameterize(mu, logvar)        
        dec = self.decoder(z.view(n_size, enc.size(1)/2, enc.size(2), enc.size(3)))
        dec = F.sigmoid(dec)
        return dec

class VAE_Loss(nn.Module):
    def __init__(self, VAE_model):
        super(VAE_Loss, self).__init__()
        assert VAE_model.__class__ == Generic_VAE, 'Only Generic_VAEs are accepted.'
        self.VAE = VAE_model
        self.size_average = True
        self.BCE = nn.BCELoss(size_average=False)
    def forward(self, X, Y):
        BCE_loss = self.BCE(X, Y)
        mu, logvar = self.VAE.last_mu, self.VAE.last_std
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (BCE_loss + KLD)/X.numel()
