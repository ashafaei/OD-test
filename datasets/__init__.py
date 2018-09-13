import os.path

import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

"""
 The data must be divided in three ways for Train, Valid, and Test.
 Therefore we implement our own wrapper around the existing datasets
 to ensure consistency across evaluations.
"""
class SubDataset(data.Dataset):
    """
        SubDataset facilitates
            (i) label override,
            (ii) second transform,
            (iii) random data splitting
        You can optionally provide a cached flag to indicate that the target process will be
        relying on cached output of the data, therefore knowing the index of the underlying data will be essential
        to retrieve from the cache. When cached is activated, the pair (x, y) becomes (x, (y, idx)).
        When optimizing for threshold, for instance, you don't need to run the underlying network for each input entry.
        Using the index, you can just fetch the cached network output. See the implementation for examples.
    """
    def __init__(self, name, parent_dataset, indices, label=None, transform=None, cached=False):
        self.parent_dataset = parent_dataset
        self.name = name
        self.indices = indices
        self.label = label
        self.transform = transform
        self.cached = cached
    
    def __len__(self):
        return self.indices.numel()
    
    def __getitem__(self, idx):
        item, label = self.parent_dataset[self.indices[idx]]

        if self.transform is not None:
            item = self.transform(item)

        output_label = label

        if self.label is not None:
            output_label = self.label
        
        if self.cached:
            output_label = (output_label, idx)

        return item, output_label

    def trim_dataset(self, new_length):
        """
            Trim the dataset by picking the first new_length entries.
        """
        assert self.indices.numel() >= new_length
        self.indices = self.indices[0:new_length]

    def split_dataset(self, p):
        """
            Randomly split the data into approximately p, 1-p sets.
        """
        p1 = torch.FloatTensor(self.indices.numel()).fill_(p).bernoulli().byte()
        d1 = SubDataset(self.name, self.parent_dataset, self.indices[p1], label=self.label,
                        transform=self.transform, cached=self.cached)
        d2 = SubDataset(self.name, self.parent_dataset, self.indices[1-p1], label=self.label,
                        transform=self.transform, cached=self.cached)
        return d1, d2

class ExpandRGBChannels(object):
    """
        This transform exapands to 3 channels if the data is not already three channels.
        Expectedly, it does not magically colorize the image!
        When MNIST is put against CIFAR, for example, we expand the channels of MNIST.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be expanded.

        Returns:
            Tensor: Expanded Tensor image.
        """
        if tensor.size(0) == 3:
            return tensor
        else:
            return torch.cat([tensor, tensor, tensor], 0)

class AbstractDomainInterface(object):
    """
        All the datasets used in this project must implement this interface.
        P.S: I really hate the way python handles inheritence and abstractions.
    """
    def __init__(self):
        self.name = self.__class__.__name__

    """
        D1's are used for the reference datasets.

        D1_train should return the class number as Y, whereas D1_valid and D1_test must
        return 0 as the class label (indicating the source distribution label rather than the class label).
    """
    def get_D1_train(self):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
    def get_D1_valid(self):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
    def get_D1_test(self):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))

    """
        D2's are used for the validation and target datasets.
        They are only used in validation (stage 2) and test (stage 3).
        We assume D1 != D2. It is strictly handled through the compatiblity function below.
        The label of Y should 1 for d2, which indicates the label of the out-of-distribution class.
    """
    def get_D2_valid(self, D1):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
    def get_D2_test(self, D1):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))    
    
    """
        This is evaluated through the lens of D2.
        Is d1 compatible with this d2?
        For Half-Classes, D1==D2, but they are still compatible because the
        D2 is the other half of the same dataset.
    """
    def is_compatible(self, D1):
        import global_vars as Global
        
        if Global.d2_compatiblity.has_key(self.name):
            return D1.name in Global.d2_compatiblity[self.name]
        else:
            raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))

    """
        Returns an image transformer that can convert other (compatible) datasets
        to this datasets for conformity in evaluation.
    """
    def conformity_transform(self):
        raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))        

class MirroredDataset(data.Dataset):
    def __init__(self, parent_dataset):
        self.parent_ds = parent_dataset
        self.length    = len(parent_dataset)
        im, _ = parent_dataset[0]
        self.pick = torch.arange(im.size(2)-1, -1, -1).long()

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        im, label = self.parent_ds[idx]
        return im[:, :, self.pick], label