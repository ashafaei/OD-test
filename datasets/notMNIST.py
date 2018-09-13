from __future__ import print_function
import os
import os.path
import errno
import torch
import torch.utils.data as data
from PIL import Image
from termcolor import colored
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface


"""
    This is a modified copy of the MNIST dataset to implement the NotMNIST dataset.
"""

class NotMNISTParent(data.Dataset):
    """`NotMNIST <http://yaroslavvb.blogspot.ca/2011/09/notmnist-dataset.html/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        assert train==True, 'There is no separate test file. Must initialize in train mode'

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if not os.path.exists(os.path.join(self.root, self.raw_folder, 'notMNIST_small.mat')):
            for url in self.urls:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())

        # process and save as torch files
        print('Processing...')
        import scipy.io as sio

        data = sio.loadmat(os.path.join(self.root, self.raw_folder, 'notMNIST_small.mat'))
        images = torch.ByteTensor(data['images']).permute(2, 0, 1) # The data is stored as HxWxN, need to permute!
        labels = torch.LongTensor(data['labels'])

        data_set = (
            images,
            labels,
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(data_set, f)

        print('Done!')

class NotMNIST(AbstractDomainInterface):
    """
        NotMNIST: 18,724 train.
        D1: not defined.
        D2: 9362 Valid, 9362 Test. (indices chosen at random)
    """

    def __init__(self):
        super(NotMNIST, self).__init__()
        
        im_transformer  = transforms.Compose([transforms.ToTensor()])
        root_path       = './workspace/datasets/notmnist'
        self.ds_train   = NotMNISTParent(root_path,
                                         train=True,
                                         transform=im_transformer,
                                         download=True)

        index_file = os.path.join('./datasets/permutation_files/', 'notmnist.pth')
        all_indices = None
        if os.path.isfile(index_file):
            all_indices = torch.load(index_file)
        else:
            print(colored('GENERATING PERMUTATION FOR NOT MNIST', 'red'))
            all_indices = torch.randperm(18724)
            torch.save(all_indices, index_file)

        self.D2_valid_ind = all_indices[0:9362]
        self.D2_test_ind  = all_indices[9362:18724]
    
    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_test_ind, label=1, transform=D1.conformity_transform())
