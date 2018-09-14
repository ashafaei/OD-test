# How to add a new

In this document we present a step-by-step guide to add a new architecture, dataset, or method.

## Architecture

In this tutorial we assume you want to add a new architecture to be used instead of the default `VGG` or `Resnet`.

You can examine the currently implemented architectures in [classifiers.py](../models/classifiers.py). You would define the architectures in the same way you do in any other PyTorch project. But you must also keep the following in mind:

- The input of your network will be in the range `[0, 1]`. If you need normalization, you must normalize the input inside your architecture.
- The forward function must have an additional argument `softmax=True`. After you get the output of your network, you should only `log_softmax` when `softmax==True`.
- You need to define an additional function `output_size(self)` that returns the output dimensionality of the network per input. For instance, the MNIST architectures return `torch.LongTensor([1, 10])`.
- You can optionally define a function `train_config` that returns a dictionary of desired configuration for the training. For example:

```python
def train_config(self):
    config = {}
    config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
    config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
    config['max_epoch'] = 60
    return config
```

After you have defined the architectures for each dataset. You must add it to the [global_variable](../global_vars.py) `dataset_reference_classifiers`. It currently says:

```python
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
```
**NOTE THAT THESE ARE CLASSES RATHER THAN INSTANCES**

After adding the new architecture, you can simply run
```bash
(env) > python setup/model_setup.py --save --exp model_ref # To train and save the model_ref in the right directory.
(env) > python setup/model_setup.py --exp model_ref # To just run the training without saving.
```
Note that `model_setup.py` trains all the architectures. If you do not have the pretrained models, or there's an architecture missing, be aware that the script would also train that architecture. If you don't wish to go through this you can change the `model_setup.py` like this:

```python
new_reference_classifiers = {
    'MNIST':                  [MNISTArch],
    'FashionMNIST':           [FMNISTArch],
}

task_list = [
    # The list of models,   The function that does the training,    Can I skip-test?,   suffix of the operation.
    # The procedures that can be skip-test are the ones that we can determine
    # whether we have done them before without instantiating the network architecture or dataset.
    # saves quite a lot of time when possible.
    # (Global.dataset_reference_classifiers, CLSetup.train_classifier,            True, ['base']),
    # (Global.dataset_reference_classifiers, KLogisticSetup.train_classifier,     True, ['KLogistic']),
    # (Global.dataset_reference_classifiers, DeepEnsembleSetup.train_classifier,  True, ['DE.%d'%i for i in range(5)]),
    # (Global.dataset_reference_autoencoders, AESetup.train_BCE_AE,               False, []),
    # (Global.dataset_reference_autoencoders, AESetup.train_MSE_AE,               False, []),
    # (Global.dataset_reference_vaes, AESetup.train_variational_autoencoder,      False, []),
    # (Global.dataset_reference_pcnns, PCNNSetup.train_pixelcnn,                  False, []),
    (new_reference_classifiers, CLSetup.train_classifier,            True, ['base']),
]
```

Here we commented out every other task, and added a new task that only includes the training of a limited set of architectures. You can track the progress of training in `visdom`.

Now, you can run the evaluation of methods that rely on classifiers with your new architecture. If the index of your architecture in the global variables is `2` running evaluation with `score_svm/2` would execute the ScoreSVM method with your architecture.

You can similarly try out different autoencoder architectures and so on ...

## Dataset

Adding a dataset is perhaps the easiest of the three. There are two types of datasets. The ones that you could use as a `D_s`, like MNIST, and the ones that you could use as `D_v, D_t`, the outliers. For instance, the noise dataset can only be used as an outlier.

To add a new dataset you must implement the AbstractDomainInterface. You can read more about it in [code organization](code_organization.md).

A dataset in this project is not the same dataset concept as in PyTorch. The datasets here are the parent objects that return datasets for each use case, like a dataset factory. The returned datasets must be an instance of `SubDataset` class, which is a simple wrapper around the PyTorch datasets. The [MNIST](../datasets/MNIST.py) implementation for instance is an easy example of how you can use an existing dataset for this project.

If your method can be used as `D_s` you must implement
```python
def get_D1_train(self):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
def get_D1_valid(self):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
def get_D1_test(self):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
def conformity_transform(self):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))        
```

For `D1_train` the `(x_i, y_i)` should be the actual underlying class. For `D1_valid` and `D1_test` however, it should be `(x_i, 0)`, 0 for the label. The `SubDataset` class allows you to easily override the label with minimum effort.

You must also implement a function that returns the transformation which would make any image compatible with the current dataset in terms of spatial size and image channels. For instance, in MNIST we return

```python
def conformity_transform(self):
    return transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((28, 28)),
                                transforms.Grayscale(),
                                transforms.ToTensor()
                                ])
```

Any outlier transformed like above would be compatible with MNIST.

If the dataset is going to be used as an outlier, you should implement the following:
```python
def get_D2_valid(self, D1):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))
def get_D2_test(self, D1):
    raise NotImplementedError("%s has no implementation for this function."%(self.__class__.__name__))    
```
The outputs are similar to `D1_valid` and `D1_test`, except the label must be 1 instead of 0. It is perhaps easier to look at an existing implementation to higlight the important steps that must be taken.

```python
def get_D2_valid(self, D1):
    assert self.is_compatible(D1)
    return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

def get_D2_test(self, D1):
    assert self.is_compatible(D1)
    return SubDataset(self.name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())
```

The input `D1` is the `D_s` for which we will be using the `self` object as an outlier. We first make sure that `self` and `D1` are compatible. For instance, if there are classes in common, we should make sure that we do not have overlapping classes. `CIFAR10` and `STL10` are for example not compatible with each other, because 9/10 classes are the same.

Then we return a `SubDataset` and transform the image according to the `conformity_transform` of the other dataset. The conformity transform ensures the outliers have the same spatial size and number of channels as `D1` samples.

After adding a new dataset, we must now add it to the [global variables](../global_vars.py) `all_dataset_classes` and `d2_compatiblity`. You can read more about this step in [code organization](code_organization.md#datasets).

Now you can run the evaluation with your newly defined dataset. Keep in mind that you must also train proper architectures for your dataset if you are running the exisitng implementations. See the previous section of this document.

## Method
