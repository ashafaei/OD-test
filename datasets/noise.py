import torch
import torch.utils.data as data
from datasets import AbstractDomainInterface

class UnifGenerator(data.Dataset):
    def __init__(self, channels, height, width, length):
        self.height   = height
        self.width    = width
        self.length   = length
        self.channels = channels
        self.name = 'UniformNoise'
        self.dataset  = torch.FloatTensor(self.length, self.channels, self.height, self.width).uniform_()

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # Unif generator always has label 1 because it comes as D2.
        return self.dataset[idx], 1
    def trim_dataset(self, new_length):
        assert self.length >= new_length
        self.dataset = self.dataset[0:new_length]
        self.length = len(self.dataset)

class GaussGenerator(data.Dataset):
    def __init__(self, channels, height, width, length):
        self.height   = height
        self.width    = width
        self.length   = length
        self.channels = channels
        self.name = 'NormalNoise'
        self.dataset  = torch.FloatTensor(self.length, self.channels, self.height, self.width).normal_(mean=0.5, std=0.25).clamp_(min=0,max=1)

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # Unif generator always has label 1 because it comes as D2.
        return self.dataset[idx], 1
    def trim_dataset(self, new_length):
        assert self.length >= new_length
        self.dataset = self.dataset[0:new_length]
        self.length = len(self.dataset)

class UniformNoise(AbstractDomainInterface):
    """
        Independent uniform random noise.
    """
    
    def get_D2_valid(self, D1):
        other_ds = D1.get_D1_valid()
        # other_train = D1.get_D1_train() # We will trim it later if necessary.
        im, _ = other_ds[0]
        length = len(other_ds)# + len(other_train)
        return UnifGenerator(im.size(0), im.size(1), im.size(2), length)

    def get_D2_test(self, D1):
        other_ds = D1.get_D1_test()
        im, _ = other_ds[0]
        length = len(other_ds)
        return UnifGenerator(im.size(0), im.size(1), im.size(2), length)

    def is_compatible(self, D1):
        return True

class NormalNoise(AbstractDomainInterface):
    """
        Independent uniform random noise.
    """
    
    def get_D2_valid(self, D1):
        other_ds = D1.get_D1_valid()
        # other_train = D1.get_D1_train() # We will trim it later if necessary.        
        im, _ = other_ds[0]
        length = len(other_ds)# + len(other_train)
        return GaussGenerator(im.size(0), im.size(1), im.size(2), length)

    def get_D2_test(self, D1):
        other_ds = D1.get_D1_test()
        im, _ = other_ds[0]
        length = len(other_ds)
        return GaussGenerator(im.size(0), im.size(1), im.size(2), length)

    def is_compatible(self, D1):
        return True