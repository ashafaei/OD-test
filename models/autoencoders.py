import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from pytorch_wavelets import DWTForward, DWTInverse

"""
    The VAE code is based on
    https://github.com/pytorch/examples/blob/master/vae/main.py
    The q(x|z)-decoder is a bernoulli distribution rather than a Gaussian.
"""

class Generic_AE(nn.Module):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256, split_size=0):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(Generic_AE, self).__init__()

        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:0')

        self.split_size = split_size

        kernel_size = 3
        all_channels = []
        current_channels = 64
        nonLin = nn.ELU
        self.epoch_factor = max(1, n_hidden/256)
        self.default_sigmoid = False
        max_pool_layers = [i%2==0 for i in range(depth)]
        remainder_layers = []
        self.netid = 'max.%d.d.%d.nH.%d'%(max_channels, depth, n_hidden)
        pad_py3 = (int)((kernel_size-1)/2)

        # encoder ###########################################
        modules = []
        in_channels = dims[0]
        in_spatial_size = dims[1]
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, padding=pad_py3))
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
        modules.append(nn.Conv2d(in_channels, n_hidden, kernel_size=kernel_size, padding=pad_py3))
        modules.append(nn.BatchNorm2d(n_hidden))
        modules.append(nonLin())
        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []
        in_channels = n_hidden
        if self.__class__ == Generic_VAE:
            in_channels = (int)(in_channels / 2)
        current_index = len(all_channels)-1
        r_ind = len(remainder_layers)-1
        for i in range(depth):
            modules.append(nn.Conv2d(in_channels, all_channels[current_index], kernel_size=kernel_size, padding=pad_py3))
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
        modules.append(nn.Conv2d(in_channels, dims[0], kernel_size=kernel_size, padding=pad_py3))
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
            sig = nn.Sigmoid()
            dec = sig(dec)
        return dec

    # because the model is split, we need to know which device the outputs go to put the labels on so the loss function can do the comparison
    def get_output_device(self):
        return self.dev2

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        #config['max_epoch'] = int(120 * self.epoch_factor)
        config['max_epoch'] = int(120)
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid

class Generic_VAE(Generic_AE):
    def __init__(self, dims, max_channels=512, depth=10, n_hidden=256, split_size=0):
        super(Generic_VAE, self).__init__(dims, max_channels, depth, 2*n_hidden, split_size)
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
        dec = self.decoder(z.view(n_size, (int)(enc.size(1)/2), enc.size(2), enc.size(3)))
        sig = nn.Sigmoid()
        dec = sig(dec)
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

# Wavelet filtered AE model
class Generic_WAE(nn.Module):
    def __init__(self, dims, levels=3, filter='db3', n_hidden=256, linear=0, split_size=0):
        assert len(dims) == 3, 'Please specify 3 values for dims'
        super(Generic_WAE, self).__init__()

        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:0')

        self.split_size = split_size

        nonLin = nn.ELU
        self.default_sigmoid = False
        self.netid = 'l.%d.f.%s.nH.%d'%(levels, filter, n_hidden)

        self.convolver = DWTForward(J=levels, mode='zero', wave=filter).to(self.dev1)
        self.devolver = DWTInverse(mode='zero',wave=filter).to(self.dev1)

        self.levels = levels
        self.channels = dims[0]

        self.start_size = dims
        self.end_size = self.calc_size_at_level(dims,levels)
        # we have (C,H,W). We end up with 4 maps (see below) so the input size to the linear layers is 4*C*H*W (per input)
        self.flat_size = 4 * self.end_size[0] * self.end_size[1] * self.end_size[2]

        # encoder ###########################################
        modules=[]

        if(linear>0):
            modules.append(nn.Linear(self.flat_size, linear))
            modules.append(nonLin())

            modules.append(nn.Linear(linear,n_hidden))
            modules.append(nonLin())
        else:
            modules.append(nn.Linear(self.flat_size, n_hidden))
            modules.append(nonLin())

        self.encoder = nn.Sequential(*modules)

        # decoder ###########################################
        modules = []

        # in reverse - start with n_hidden and go to end_size

        if(linear>0):
            modules.append(nn.Linear(n_hidden,linear))
            modules.append(nonLin())

            modules.append(nn.Linear(linear,self.flat_size))
            modules.append(nonLin())
        else:
            modules.append(nn.Linear(n_hidden, self.flat_size))
            modules.append(nonLin())
        
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        n_samples = x.size(0)
        Yl, Yh = self.convolver(x)
        print("--Encoder--")
        print("Yl Shape:" + str(Yl.shape))
        print("Yh Length:" + str(len(Yh)))
        for i in range(len(Yh)):
            print("Yh[" + str(i) + "] shape:" + str(Yh[i].shape))

        #Yl contains the approximated version(LL). Yh contains progressively smaller versions of horizontal(LH), vertical(HL) and diagonal(HH) detail tensors.
        #We take the last version from the stack, so reduce the detail by increasing 'levels', which should also reduce size of Yl
        #shape of Yl is (N,C,H,W), while shape of Yh[-1] (lat detail layer) is (N,C,3,H,W). We add these into (N,C,4,H,W) and flatten it to feed into the FC layer
        Yd = Yh[-1]
        Yl = torch.unsqueeze(Yl,2) # change to (N,C,1,H,W)
        Yd = torch.cat((Yd,Yl),2) # add together into (N,C,4,H,W)
        Yy = torch.flatten(Yd, start_dim=1) # flatten to (N,C*4*H*W)

        code = self.encoder(Yy) # encode to (N,n_hidden)
        out = code.view(n_samples, -1) # flatten to vectors.
        return out

    def decode(self,x):
        # should start with N vectors of n_hidden encodings
        n_samples = x.size(0)
        Yy = self.decoder(x) # comes out as (N,C*4*H*W)
        Yd = torch.reshape(Yy ,(n_samples, self.channels, 4, self.end_size[1], self.end_size[2])) # reshape to (N,C,4,H,W)
        Yd,Yl = torch.split(Yd,[3,1],dim=2) # split apart into (N,C,3,H,W) (Yd) and (N,C,1,H,W) (Yl)
        Yl = torch.squeeze(Yl,2) # resqueeze Yl to (N,C,H,W)
        Yh = self.build_coeff_at_detail_level(self.start_size, n_samples, Yd, self.levels) # build estimated coefficient tree
        print("--Decoder--")
        print("Yl Shape:" + str(Yl.shape))
        print("Yd Shape:" + str(Yd.shape))
        print("Yh Length:" + str(len(Yh)))
        for i in range(len(Yh)):
            print("Yh[" + str(i) + "] shape:" + str(Yh[i].shape))

        #now we have Yl and an estimated Yh, so feed that back into the DWT
        Xd = self.devolver((Yl,Yh))
        return Xd

    def forward(self, x, sigmoid=False):
        enc = self.encode(x)
        dec = self.decode(enc)
        if sigmoid or self.default_sigmoid:
            sig = nn.Sigmoid()
            dec = sig(dec)
        return dec

    def calc_size_at_level(self,input_size,level):
        #expects the last two values to be H and W, retains all others
        # only works for DWT
        out = list(input_size)
        for i in range(level):
            W = out[-1] + 4
            H = out[-2] + 4
            W = math.ceil(W / 2)
            H = math.ceil(H / 2)
            out[-1] = W
            out[-2] = H
        return tuple(out)

    def build_coeff_at_detail_level(self,input_size,samples,input_coeff,level):
        out = []

        #we want a list of tensors. Input size is original input size - ie, original image size. We expand this to include batch, channel and filter
        #for MNIST, 1,28,28 -> 64,1,3,28,28

        base_size = torch.Size((samples,self.channels,3,input_size[1],input_size[2]))

        for i in range(level):
            if(i==level-1):
                out.append(input_coeff)
            else:
                size = self.calc_size_at_level(base_size,i+1)
                out.append(torch.zeros(size).to(self.dev1))
        return out

    # because the model is split, we need to know which device the outputs go to put the labels on so the loss function can do the comparison
    def get_output_device(self):
        return self.dev2

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
        #config['max_epoch'] = int(120 * self.epoch_factor)
        config['max_epoch'] = int(120)
        return config

    def preferred_name(self):
        return self.__class__.__name__+"."+self.netid