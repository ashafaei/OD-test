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

    def __init__(self, drop_class=None):
        super(CIFAR10, self).__init__(drop_class = drop_class)
        
        im_transformer  = transforms.Compose([transforms.ToTensor()])
        root_path       = './workspace/datasets/cifar10'

        self.ds_train   = datasets.CIFAR10(root_path,
                                        train=True,
                                        transform=im_transformer,
                                        download=True)
        self.ds_test    = datasets.CIFAR10(root_path,
                                        train=False,
                                        transform=im_transformer,
                                        download=True)

        
        self.D1_train_ind = torch.arange(0, 40000).int()
        self.D1_valid_ind = torch.arange(40000, 50000).int()
        self.D1_test_ind  = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 50000).int()
        self.D2_test_ind  = torch.arange(0, 10000).int()

    def get_D1_train(self):
        target_indices = self.D1_train_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices)

    def get_D1_train_dropped(self):
        target_indices = self.D1_train_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name], True)
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=1)

    def get_D1_valid(self):
        target_indices = self.D1_valid_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=0)

    def get_D1_valid_dropped(self):
        target_indices = self.D1_valid_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name], True)
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=1)

    def get_D1_test(self):
        target_indices = self.D1_test_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_test, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_test, target_indices, label=0)

    def get_D1_test_dropped(self):
        target_indices = self.D1_test_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_test, target_indices, self.filter_rules[self.base_name],True)
        return SubDataset(self.name, self.base_name, self.ds_test, target_indices, label=1)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.base_name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def get_num_classes(self):
        classes = 10
        if self.name in self.filter_rules:
            dropped_classes = len(self.filter_rules[self.base_name])
            classes = classes - dropped_classes
        return classes

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

        self.filter_rules['TinyImagenet'] = [6, 21, 24, 43, 51, 53, 61, 77, 89]      
    
    def get_D1_train(self):
        target_indices = self.D1_train_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices)

    def get_D1_train_dropped(self):
        target_indices = self.D1_train_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name], True)
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=1)

    def get_D1_valid(self):
        target_indices = self.D1_valid_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=0)

    def get_D1_valid_dropped(self):
        target_indices = self.D1_valid_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[self.base_name], True)
        return SubDataset(self.name, self.base_name, self.ds_train, target_indices, label=1)

    def get_D1_test(self):
        target_indices = self.D1_test_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_test, target_indices, self.filter_rules[self.base_name])
        return SubDataset(self.name, self.base_name, self.ds_test, target_indices, label=0)

    def get_D1_test_dropped(self):
        target_indices = self.D1_test_ind
        if self.base_name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_test, target_indices, self.filter_rules[self.base_name],True)
        return SubDataset(self.name, self.base_name, self.ds_test, target_indices, label=1)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_valid_ind
        if D1.name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_train, target_indices, self.filter_rules[D1.name])
        return SubDataset(self.name, self.ds_train, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        if D1.name in self.filter_rules:
            target_indices = self.filter_indices(self.ds_test, target_indices, self.filter_rules[D1.name])
        return SubDataset(self.name, self.ds_test, target_indices, label=1, transform=D1.conformity_transform())

    def get_num_classes(self):
        classes = 100
        if self.name in self.filter_rules:
            dropped_classes = len(self.filter_rules[self.base_name])
            classes = classes - dropped_classes
        return classes

    def conformity_transform(self):
        return transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),                                   
                                   ])
