# Code Organization

- [Code Structure](#code-structure)
- [Global Variables](#global-variables)
- [Important Classes](#important-classes)

## Code Structure

- `<od-test>`
    - **setup** this folder contains all the setup scripts.
        - `setup.py` the script to setup the environment, the initial setup of the project.
        - `model_setup.py` to do the initial reference model training before the evaluation. See the [info](train_reference_models.md).
        - **categories** contains the training scripts for each category of model used during the evaluation.
    - **utils** this folder contains the helper classes that we use.
        - `args.py` the script responsible for argument parsing and environment experiment setup.
        - `iterative_trainer.py` is the iterative training class that handles various training scenarios.
        - `logger.py` is the logging helper class.
        - `experiment_merger.py` merges the *results* of multiple experiments into a single experiment. Useful for figure generation.
    - **datasets** has the classes for the datasets that we use.
        - `init.py` has some important base definitions that we need for the datasets.
            - `SubDataset` is the dataset wrapper that we use to facilitate data splitting and several other useful operations such as on-air modification of the labels or additional preprocessing steps.
            - `AbstractDomainInterface` is the dataset interface that each dataset class must implement. You can read more about it in [Important Classes](#important-classes).
            - `MirroredDataset` this class wrapper mirrors the images of another dataset.
        - `<dataset_name>.py` the implementation of individual datasets. All the classes download and preprocess the dataset automatically upon first instantiation. The downloaded files are stored in `workspace/datasets/<dataset_name>`. All the classes implement the `AbstractDomainInterface`.

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

## Important Classes

### AbstractDomainInterface
