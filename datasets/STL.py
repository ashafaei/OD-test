import torch
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
from torchvision import datasets

class STL10(AbstractDomainInterface):
    """
        STL10: 5,000 train + 8,000 test. (3x96x96)
        D1:    5,000 train + (4,000 valid + 4,000 test)
        D2:    5,000 valid + 8,000 test.
    """

    def __init__(self, downsample=None):
        super(STL10, self).__init__()

        im_transformer = None
        self.downsample = downsample
        if self.downsample is None:
            im_transformer  = transforms.Compose([transforms.ToTensor()])
        else:
            im_transformer  = transforms.Compose([transforms.Resize((self.downsample, self.downsample)), transforms.ToTensor()])
        root_path       = './workspace/datasets/stl10'
        self.D1_train_ind = torch.arange(0, 5000).int()
        self.D1_valid_ind = torch.arange(0, 4000).int()
        self.D1_test_ind  = torch.arange(4000, 8000).int()

        self.D2_valid_ind = torch.arange(0,5000).int()
        self.D2_test_ind  = torch.arange(0,8000).int()

        self.ds_train   = datasets.STL10(root_path,
                                        split='train',
                                        transform=im_transformer,
                                        download=True)
        self.ds_test    = datasets.STL10(root_path,
                                        split='test',
                                        transform=im_transformer,
                                        download=True)
    
    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_test, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        target = 96
        if self.downsample is not None:
            target = self.downsample
        out_transform = transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((target, target)),
                                   transforms.ToTensor(),
                                   ])
        return out_transform

class STL10d32(STL10):
    def __init__(self):
        super(STL10d32, self).__init__(downsample=32)
