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
    - **models** contains all the model definitions we use.
        - `classifiers.py` contains all the predictive models.
        - `autoencoders.py` contains all the (variational) autoencoder models.
        - **pixelcnn** contains the PixelCNN++ implementations taken from https://github.com/pclucas14/pixel-cnn-pp
    - **methods** contains all the methods.
        - `init.py` has some important base definitions that we need for the methods.
            - `AbstractMethodInterface` is the method interfae that each method must implement. You can read more about it in [Important Classes](#important-classes).
            - `AbstractModelWrapper` is the wrapper class used to abstract away the underlying model used in the operations. You do not have to use it for your own methods, but it should simplify some of the tasks.
            - `<method_name>.py` the implementation of each method. All the classes implement the `AbstractMethodInterface`.
    - **workspace** contains every result and output of the framework. You must set up the project for this folder to appear.
        - **datasets** will be where all the downloaded datasets are stored.
        - **env** will containt the virtual environment within which the code would work.
        - **visdom** is the default visdom home.
        - **experiments** is the location of all the experiments.
            - **model_ref** is where the reference models will be stored.
            - `<exp-name>` will be the home directory to each experiment.

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

Here is the description of the important classes that you should be aware of.

### AbstractDomainInterface
This is the dataset interface that each dataset must implement. You can examine sample implementations such as [MNIST](../datasets/MNIST.py) to better familiarize yourself with the usage. The following definition is in [datasets/init.py](../datasets/__init__.py)

```python
class AbstractDomainInterface(object):
    def get_D1_train(self):  # required
    def get_D1_valid(self):  # required
    def get_D1_test(self):   # required

    def get_D2_valid(self, D1):  # required
    def get_D2_test(self, D1):   # required

    def is_compatible(self, D1): # optional
    def conformity_transform(self): # required
```
`get_D1_{train, valid, test}` must return the split of the dataset for train, valid, and test when used as `D_s`.

`get_D2_{valid, test}` must return the split of the dataset for valid, and test when used as `D_v` or `D_t`. We also provide the `D_s` being used so that the method can take the necessary actions for spatial resizing and label filtering if necessary.

The output of these functions must be a `SubDataset` class. `SubDataset` is a dataset wrapper that simplifies several tasks. You should check out the implementation from [here](../datasets/__init__.py) for more information.

`is_compatible` says whether the provided `D_s` is compatible with this object as `D_v` or `D_t`. There's a default implementation here that relies on the compatibility table in [global_vars](../global_vars.py).

`conformity_transform` must return a series of transformations that would make other datasets compatible with this dataset. For instance, for MNIST we have

```python
return transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((28, 28)),
                            transforms.Grayscale(),
                            transforms.ToTensor()
                            ])
```
Which would resize every input to 28 x 28 and make the images grayscale. This is used in `get_D2_{valid, test}` where we wish to return a dataset that is compatible with `D_s`.

### AbstractMethodInterface
This is the method interface that each method must implement for the evaluation. The `ProbabilityThreshold` class in [base_threshold.py](../methods/base_threshold.py) is a simple example to learn more.

The `AbstractMethodInterface` is defined in [methods/init.py](../methods/__init__.py). The structure is fairly simple.

```python
class AbstractMethodInterface(object):
    def propose_H(self, dataset): # Step 1

    def train_H(self, dataset): # Step 2

    def test_H(self, dataset): # Step 3
    
    def method_identifier(self): # A string that identifies the model.
```

The input of `Step 1` is a classification dataset. Each item in the dataset is the image and its corresponding label `(x_i, y_i)`. This is the Line 3 of Algorithm 1 in the paper. You must use the given dataset in any way you need. For example:

- PBThreshold: uses this dataset to train a classifier. The classifier is then saved for the next step where the optimal threshold must be identified.
- K-NNSVM: Simply saves every `x` in the dataset as the reference within which we will later find the nearest neighbours.
- AEThreshold: Trains an autoencoder on the dataset and saves the model.

The function does not have to return anything, but then it must be ready to process `train_H`, which is the next step in the pipeline.

The input of `Step 2` is a mixture dataset for binary classification: `{D_s:0, D_t:1}`. Your method in this step must train a binary classifier. The output of the function should be the train error. The method should save the reject function inside the object and be ready to evaluate the reject function in `test_H`.

- PBThreshold: learns the optimal threshold on the output of the predicitve neural network trained in the previous step.
- K-NNSVM: trains an SVM on the sorted Euclidean distance of samples from the reference set.
- AEThreshold: learn the threshold for reconstruction error to separate the outliers.

The input of `Step 3` is a mixture dataset for binary classification, just like `Step 2`. In this step, you must evaluate the performance of the learned reject function on the given dataset and return its accuracy.

### AbstractModelWrapper
This is a helper wrapper. Your method does not have to use this. But then you may want to. Sometimes, the method depends on the output of a previously trained network. For instance, PBThreshold is thresholding the max probability of some neural network. (i) when we are learning the parameters, we only want to tune the threshold parameters and not change the underlying network, (ii) when we are doing the iterative training, we do not have to pass the input image into the network on each iteration, because the output of the underlying network does not change and it would be much slower to do this every time. This wrapper class is meant to simplify the process by implementing the general parts and allowing you to focus on the part of the implementation that is more relevant to your method.

For instance, in PBThreshold we have:

```python
class PTModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        For Base Threshold, we simply apply f(x) = sign(t-x), where x is the max probability and t is the threshold.
        We learn the the threshold with an SVM loss and a zero margin. This yields exactly the optimal threshold learning problem.
    """
    def __init__(self, base_model):
        super(PTModelWrapper, self).__init__(base_model)
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0.5]))) # initialize to prob=0.5 for faster convergence.

    """
        This function implements the part of the procedure where you have to retrieve
        the output of a subnetwork. For PBThreshold, we simply need the max probability.
        During the optimization of the threshold, the method would cache these valus. 
    """
    def subnetwork_eval(self, x):
        base_output = self.base_model(x)
        # Get the max probability out
        input = base_output.exp().max(1)[0].unsqueeze_(1)
        return input.detach()

    """
        Now, given the output of subnetwork_eval, how would you process the result.
        For PBThreshold, it is only a threshold operation. The inputs are max probabilities.
    """
    def wrapper_eval(self, x):
        # Threshold hold the max probability.
        output = self.H.threshold - x
        return output
    
    """
        Given the output of wrapper_eval, how would you classify?
        For PBThreshold, it's simply the sign. The output must be long.
    """
    def classify(self, x):
        return (x > 0).long()
```
