"""
    This file lists all the global variables that are used throughout the project.
    The two major components of this file are the list of the datasets and the list of the models.
"""

"""
    This is where we keep a reference to all the dataset classes in the project.
"""
import datasets.MNIST as MNIST
import datasets.FashionMNIST as FMNIST
import datasets.notMNIST as NMNIST
import datasets.CIFAR as CIFAR
import datasets.noise as noise
import datasets.STL as STL
import datasets.TinyImagenet as TI

all_dataset_classes = [ MNIST.MNIST, FMNIST.FashionMNIST, NMNIST.NotMNIST,
                        CIFAR.CIFAR10, CIFAR.CIFAR100,
                        STL.STL10, TI.TinyImagenet,
                        noise.UniformNoise, noise.NormalNoise,
                        STL.STL10d32, TI.TinyImagenetd32]

"""
    Not all the datasets can be used as a Dv, Dt (aka D2) for each dataset.
    The list below specifies which datasets can be used as the D2 for the other datasets.
    For instance, STL10 and CIFAR10 cannot face each other because they have 9 out 10 classes
    in common.
"""
d2_compatiblity = {
    # This can be used as d2 for            # this
    'MNIST'                                 : ['FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
    'NotMNIST'                              : ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
    'FashionMNIST'                          : ['MNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
    'CIFAR10'                               : ['MNIST', 'FashionMNIST', 'CIFAR100', 'TinyImagenet', 'TinyImagenetd32'],
    'CIFAR100'                              : ['MNIST', 'FashionMNIST', 'CIFAR10', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'],
    'STL10'                                 : ['MNIST', 'FashionMNIST', 'CIFAR100', 'TinyImagenet', 'TinyImagenetd32'],
    'TinyImagenet'                          : ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'STL10d32'],
    # STL10 is not compatible with CIFAR10 because of the 9-overlapping classes.
    # Erring on the side of caution.
}

# We can augment the following training data with mirroring.
# We make sure there's no information leak in-between tasks.
mirror_augment = {
    'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet', 'STL10d32', 'TinyImagenetd32'
}

"""
    This where we keep a reference to all the models in the project.
"""

import models.classifiers as CLS
import models.autoencoders as AES
import models.pixelcnn.model as PCNN

"""
    Each dataset has a list of compatible neural netwok architectures.
    Your life would be simpler if you keep the same family as the same index within each dataset.
    For instance, VGGs are all 0 and Resnets are all 1.
"""
dataset_reference_classifiers = {
    'MNIST':                  [CLS.MNIST_VGG,         CLS.MNIST_Resnet],
    'FashionMNIST':           [CLS.MNIST_VGG,         CLS.MNIST_Resnet],
    'CIFAR10':                [CLS.CIFAR10_VGG,       CLS.CIFAR10_Resnet],
    'CIFAR100':               [CLS.CIFAR100_VGG,      CLS.CIFAR100_Resnet],
    'STL10':                  [CLS.STL10_VGG,         CLS.STL10_Resnet],
    'TinyImagenet':           [CLS.TinyImagenet_VGG,  CLS.TinyImagenet_Resnet],
}

class ModelFactory(object):
    def __init__(self, parent_class, **kwargs):
        self.parent_class = parent_class
        self.kwargs = kwargs
    def __call__(self):
        return self.parent_class(**self.kwargs)

dataset_reference_autoencoders = {
    'MNIST':              [ModelFactory(AES.Generic_AE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
    'FashionMNIST':       [ModelFactory(AES.Generic_AE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
    'CIFAR10':            [ModelFactory(AES.Generic_AE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
    'CIFAR100':           [ModelFactory(AES.Generic_AE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
    'STL10':              [ModelFactory(AES.Generic_AE, dims=(3, 96, 96), max_channels=512, depth=12, n_hidden=512)],
    'TinyImagenet':       [ModelFactory(AES.Generic_AE, dims=(3, 64, 64), max_channels=512, depth=12, n_hidden=512)],
}

dataset_reference_vaes = {
    'MNIST':              [ModelFactory(AES.Generic_VAE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
    'FashionMNIST':       [ModelFactory(AES.Generic_VAE, dims=(1, 28, 28), max_channels=256, depth=8, n_hidden=96)],
    'CIFAR10':            [ModelFactory(AES.Generic_VAE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
    'CIFAR100':           [ModelFactory(AES.Generic_VAE, dims=(3, 32, 32), max_channels=512, depth=10, n_hidden=256)],
    'STL10':              [ModelFactory(AES.Generic_VAE, dims=(3, 96, 96), max_channels=512, depth=12, n_hidden=512)],
    'TinyImagenet':       [ModelFactory(AES.Generic_VAE, dims=(3, 64, 64), max_channels=512, depth=12, n_hidden=512)],
}

dataset_reference_pcnns = {
    'MNIST':              [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=32, input_channels=1, nr_logistic_mix=5)],
    'FashionMNIST':       [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=64, input_channels=1, nr_logistic_mix=5)],
    'CIFAR10':            [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)],
    'CIFAR100':           [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)],
    'TinyImagenetd32':    [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)],
    'STL10d32':           [ModelFactory(PCNN.PixelCNN, nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)],
}

"""
    This is where we keep a reference to all the methods.
"""

import methods.base_threshold as BT
import methods.score_svm as SSVM
import methods.logistic_threshold as KL
import methods.mcdropout as MCD
import methods.nearest_neighbor as KNN
import methods.binary_classifier as BinClass
import methods.deep_ensemble as DE
import methods.odin as ODIN
import methods.reconstruction_error as RE
import methods.pixelcnn as PCNN
import methods.openmax as OM

all_methods = {
    'prob_threshold':   BT.ProbabilityThreshold,
    'score_svm':        SSVM.ScoreSVM,
    'logistic_svm':     KL.LogisticSVM,
    'mcdropout':        MCD.MCDropout,
    'knn':              KNN.KNNSVM,
    'bceaeknn':         KNN.BCEKNNSVM,
    'mseaeknn':         KNN.MSEKNNSVM,
    'vaeaeknn':         KNN.VAEKNNSVM,
    'binclass':         BinClass.BinaryClassifier,
    'deep_ensemble':    DE.DeepEnsemble,
    'odin':             ODIN.ODIN,
    'reconst_thresh':   RE.ReconstructionThreshold,
    'pixelcnn':         PCNN.PixelCNN,
    'openmax':          OM.OpenMax,
}

##################################################################
# Do not change anything below, unless you know what you are doing.
"""
    all_datasets is automatically generated
    all_datasets = {
        'MNIST' : MNIST,
        ...
    }
    
"""
all_datasets = {}
for dscls in all_dataset_classes:
    all_datasets[dscls.__name__] = dscls

def get_ref_classifier(dataset):
    if dataset_reference_classifiers.has_key(dataset):
        return dataset_reference_classifiers[dataset]
    raise NotImplementedError()

def get_ref_autoencoder(dataset):
    if dataset_reference_autoencoders.has_key(dataset):
        return dataset_reference_autoencoders[dataset]
    raise NotImplementedError()

def get_ref_vae(dataset):
    if dataset_reference_vaes.has_key(dataset):
        return dataset_reference_vaes[dataset]
    raise NotImplementedError()

def get_ref_pixelcnn(dataset):
    if dataset_reference_pcnns.has_key(dataset):
        return dataset_reference_pcnns[dataset]
    raise NotImplementedError()

def get_method(name, args):
    elements = name.split('/')
    instance = all_methods[elements[0]](args)
    if len(elements) > 1:
        instance.default_model = int(elements[1])
    return instance
    