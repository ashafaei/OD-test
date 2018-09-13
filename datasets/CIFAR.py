import torch
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
from torchvision import datasets

def filter_indices(dataset, indices, filter_label):
    accept = []
    for ind in indices:
        _, label = dataset[ind]
        if label not in filter_label:
            accept.append(ind)
    return torch.IntTensor(accept)

class CIFAR10(AbstractDomainInterface):
    """
        CIFAR10: 50,000 train + 10,000 test. (3x32x32)
        D1: (40,000 train + 10,000 valid) + (10,000 test)
        D2 (Dv, Dt): 50,000 valid + 10,000 test.
    """

    def __init__(self):
        super(CIFAR10, self).__init__()
        
        im_transformer  = transforms.Compose([transforms.ToTensor()])
        root_path       = './workspace/datasets/cifar10'
        self.D1_train_ind = torch.arange(0, 40000).int()
        self.D1_valid_ind = torch.arange(40000, 50000).int()
        self.D1_test_ind  = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 50000).int()
        self.D2_test_ind  = torch.arange(0, 10000).int()

        self.ds_train   = datasets.CIFAR10(root_path,
                                        train=True,
                                        transform=im_transformer,
                                        download=True)
        self.ds_test    = datasets.CIFAR10(root_path,
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
        return transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),                                   
                                   ])

class CIFAR100(AbstractDomainInterface):
    """
        CIFAR100: 50,000 train + 10,000 test. (3x32x32)
        D1: (40,000 train + 10,000 valid) + (10,000 test)
        D2 (Dv , Dt): 50,000 valid + 10,000 test.
    """

    def __init__(self):
        super(CIFAR100, self).__init__()
        
        im_transformer  = transforms.Compose([transforms.ToTensor()])
        root_path       = './workspace/datasets/cifar100'
        self.D1_train_ind = torch.arange(0, 40000).int()
        self.D1_valid_ind = torch.arange(40000, 50000).int()
        self.D1_test_ind  = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 50000).int()
        self.D2_test_ind  = torch.arange(0, 10000).int()

        self.ds_train   = datasets.CIFAR100(root_path,
                                        train=True,
                                        transform=im_transformer,
                                        download=True)
        self.ds_test    = datasets.CIFAR100(root_path,
                                        train=False,
                                        transform=im_transformer,
                                        download=True)

        """
            TinyImagenet:
                 6:bee with 38:bee
                 21:chimpanzee with 55:chimpanzee, chimp, Pan troglodytes
                 24:cockroach with 41:cockroach, roach
                 43:lion with 34:lion, king of beasts, Panthera leo
                 51:mushroom with 185:mushroom
                 53:orange with 186:orange
                 61:plate with 177:plate
                 77:snail with 15:snail
                 89:tractor with 164:tractor
        """
        self.filter_rules = {
            'TinyImagenet': [6, 21, 24, 43, 51, 53, 61, 77, 89]
        }                                        
    
    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_train, self.D1_valid_ind, label=0)
    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_valid_ind
        if self.filter_rules.has_key(D1.name):
            target_indices = filter_indices(self.ds_train, target_indices, self.filter_rules[D1.name])
        return SubDataset(self.name, self.ds_train, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        if self.filter_rules.has_key(D1.name):
            target_indices = filter_indices(self.ds_test, target_indices, self.filter_rules[D1.name])
        return SubDataset(self.name, self.ds_test, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),                                   
                                   ])                               
