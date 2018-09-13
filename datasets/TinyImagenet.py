from __future__ import print_function
import os
import os.path
import errno
import torch
import torch.utils.data as data
from PIL import Image
from termcolor import colored
import torchvision.transforms as transforms
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels


"""
    This is a modified copy of the MNIST+ImageFolderDataset dataset to implement the TinyImagenet dataset.
"""
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, suffix=''):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if suffix != '':
            d = os.path.join(d, suffix)

        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class TinyImagenetParent(data.Dataset):
    """`TinyImagenet <https://tiny-imagenet.herokuapp.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where the data exist.
        split (string, optional): {'train', 'val', 'test'}.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'index.pt'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        assert split in ['train', 'test', 'valid']

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train, test or valid

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        
        self.index = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        self.key = split

        # if self.train:
        #     self.train_data, self.train_labels = torch.load(
        #         os.path.join(self.root, self.processed_folder, self.training_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        im_path, target = self.index[self.key][index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = default_loader(im_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index[self.key])

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file))

    def download(self):
        """Download the TinyImagenet data if it doesn't exist in processed_folder already."""
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

        if not os.path.exists(os.path.join(self.root, self.raw_folder, 'tiny-imagenet-200.zip')):
            for url in self.urls:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())
        extract_path = os.path.join(self.root, self.raw_folder)
        output_path = os.path.join(extract_path, 'tiny-imagenet-200')
        if not os.path.isdir(output_path):
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(extract_path)
        
        index_path = os.path.join(self.root, self.processed_folder, self.training_file)

        if not os.path.exists(index_path):
            # process and save as torch files
            print('Processing...')

            train_path = os.path.join(output_path, 'train')
            classes, class_to_idx = find_classes(train_path)
            train_set = make_dataset(train_path, class_to_idx)
            valid_set = []
            with open(os.path.join(output_path, 'val', 'val_annotations.txt'), 'r') as f:
                for row in f:
                    row = row.split('\t')
                    valid_set.append((os.path.join(output_path, 'val', 'images', row[0]), class_to_idx[row[1]]))
            test_set = make_dataset(os.path.join(output_path, 'test'), {'images':-1})

            torch.save({'class_to_index': class_to_idx, 'train': train_set, 'valid': valid_set, 'test': test_set}, index_path)
            print('Done!')

def filter_indices(dataset, indices, filter_label):
    accept = []
    for ind in indices:
        _, label = dataset.index[dataset.key][ind]
        if label not in filter_label:
            accept.append(ind)
    return torch.IntTensor(accept)


class TinyImagenet(AbstractDomainInterface):
    """
        TinyImagenet: 100,000 train, 10,000 valid, 10,000 test.
        D1: 100,000 train (shuffled), 10,000 valid, 10,000 test.
        D2: 10,000 Valid + 100,000 train (shuffled), 10,000 Test.
    """

    def __init__(self, downsample=None):
        super(TinyImagenet, self).__init__()
        
        im_transformer = None
        self.downsample = downsample
        if self.downsample is None:
            im_transformer  = transforms.Compose([transforms.ToTensor()])
        else:
            im_transformer  = transforms.Compose([transforms.Resize((self.downsample, self.downsample)), transforms.ToTensor()])
        root_path       = './workspace/datasets/tinyimagenet'
        self.ds_train   = TinyImagenetParent(root_path,
                                            split='train',
                                            transform=im_transformer,
                                            download=True)
        self.ds_valid   = TinyImagenetParent(root_path,
                                            split='valid',
                                            transform=im_transformer,
                                            download=True)
        self.ds_test    = TinyImagenetParent(root_path,
                                            split='test',
                                            transform=im_transformer,
                                            download=True)

        index_file = os.path.join('./datasets/permutation_files/', 'tinyimagenet.pth')
        train_indices = None
        if os.path.isfile(index_file):
            train_indices = torch.load(index_file)
        else:
            print(colored('GENERATING PERMUTATION FOR <TinyImagenet train>', 'red'))
            train_indices = torch.randperm(len(self.ds_train))
            torch.save(train_indices, index_file)

        self.D1_train_ind = train_indices.int()
        self.D1_valid_ind = torch.arange(0, len(self.ds_valid)).int()
        self.D1_test_ind  = torch.arange(0, len(self.ds_test)).int()

        self.D2_valid_ind = train_indices.int()
        self.D2_test_ind  = torch.arange(0, len(self.ds_valid)).int()

        """
        CIFAR100:
                 15:snail with 77:snail
                 34:lion, king of beasts, Panthera leo with 43:lion
                 38:bee with 6:bee
                 41:cockroach, roach with 24:cockroach
                 55:chimpanzee, chimp, Pan troglodytes with 21:chimpanzee
                 164:tractor with 89:tractor
                 177:plate with 61:plate
                 185:mushroom with 51:mushroom
                 186:orange with 53:orange
        """
        self.filter_rules = {
            'CIFAR100': [15, 34, 38, 41, 55, 164, 177, 185, 186]
        }

    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)
    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_valid, self.D1_valid_ind, label=0)
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
            target_indices = filter_indices(self.ds_valid, target_indices, self.filter_rules[D1.name])
        return SubDataset(self.name, self.ds_valid, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        target = 64
        if self.downsample is not None:
            target = self.downsample
        out_transform = transforms.Compose([ExpandRGBChannels(),
                                   transforms.ToPILImage(),
                                   transforms.Resize((target, target)),
                                   transforms.ToTensor(),
                                   ])
        return out_transform

class TinyImagenetd32(TinyImagenet):
    def __init__(self):
        super(TinyImagenetd32, self).__init__(downsample=32)
