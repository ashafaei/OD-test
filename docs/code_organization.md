# Code Organization

- [Global Variables](#global-variables)

## Global Variables
There are a few global variables that we use throughout the project. You can examine [global_vars.py](../global_vars.py) to get an idea. In this file, we include the list of the available **datasets**, **network architectures**, and **methods** to be used. If you wish to add a new dataset, network architecture, or method you must remember to add the necessary information in the `global_vars.py`.

### Datasets
For datasets, there are two important variables: (i) `all_dataset_classes`, and (ii) `d2_compatiblity`. The first variable is simply a list of the dataset _classess_. To add a new dataset, you must implement the `AbstractDomainInterface`. See the existing implementations to see how it works, or read the tutorial on that (the link is on the first page).

The second variable contains the compatibility table. Not all the datasets can be used as D_v and D_t for every dataset. For instance, the `STL10` has 9/10 classes in common with `CIFAR10`, therefore, we do not ever use `D2=STL10` and `D1=CIFAR10` or vice-versa. In the example below, the compatibility table says that when `D1=FashionMNIST` or `D1=CIFAR10`, we can use `D2=MNIST`.

```python
d2_compatiblity = {
    # This can be used as d2 for            # this
    'MNIST'                                 : ['FashionMNIST', 'CIFAR10'],
}
```

### Network Architectures
The network architectures are organized by their purpose. Some architectures are for classification, some for autoencoding, etc.

For classification, the variable is `dataset_reference_classifiers`. For each dataset, you must include a list of compatible classifier architecture. If you intend to add a family of architectures, we recommend that you ensure all the instances of that family have the same index within this table. An example table is below:

```python
dataset_reference_classifiers = {
    'MNIST':                [CLS.MNIST_VGG,         CLS.MNIST_Resnet],
}
```

### Methods
The methods section simply contains a dictionary of all the implemented methods.

```python
all_methods = {
    'prob_threshold':   BT.ProbabilityThreshold,
    'score_svm':        SSVM.ScoreSVM,
    ...
}
```
For more information on the implemented methods see the first page.
