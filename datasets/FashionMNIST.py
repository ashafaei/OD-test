import torch
import torchvision.transforms as transforms
from torchvision import datasets

from datasets import SubDataset, AbstractDomainInterface

class FashionMNIST(AbstractDomainInterface):
    """
        FashionMNIST: 60,000 train + 10,000 test.
        D1: (50,000 train + 10,000 valid) + (10,000 test)
        D2: 60,000 valid + 10,000 test.
    """

    def __init__(self):
        super(FashionMNIST, self).__init__()

        im_transformer  = transforms.Compose([transforms.ToTensor()])
        root_path       = './workspace/datasets/fmnist'
        self.D1_train_ind = torch.arange(0, 50000).int()
        self.D1_valid_ind = torch.arange(50000, 60000).int()
        self.D1_test_ind  = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 60000).int()
        self.D2_test_ind  = torch.arange(0, 10000).int()

        self.ds_train   = datasets.FashionMNIST(root_path,
                                        train=True,
                                        transform=im_transformer,
                                        download=True)
        self.ds_test    = datasets.FashionMNIST(root_path,
                                        train=False,
                                        transform=im_transformer,
                                        download=True)
    
    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_train, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((28, 28)),
                                   transforms.Grayscale(),
                                   transforms.ToTensor()
                                   ])
