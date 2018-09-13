# Setting up the Reference Models
Most of the methods rely on some underlying model to make the rejct decision for `OD-test`. For instance, `PbThreshold` would threshold the maximum probability of the output of a discriminative neural network to reject out-of-distribution samples. Therefore, we need to provide a set of pre-trained reference models on which these methods would operate. In the original paper we evaluated the performance on `ResNet-50` and `VGG-16` variant networks. Note that each dataset would have its own pre-trained model, and also the architectures would slightly vary between the datasets as the spatial dimension and the number of classes of each dataset is different.

Since the training of the reference models is time consuming, we separately train all the reference models before running the evaluation. You can alternatively download the pretrained reference models from [here](pretrained.md). The script that takes care of this time-consuming training is [model_setup.py](../setup/model_setup.py).

To train all the reference models run the following command:

```bash
> cd <root_folder> # Go to the root folder.
> source workspace/env/bin/activate # Make sure the environment is active.
(env)> python setup/model_setup.py --exp model_ref --save
```

 **Warning this will take a loooong time to finish.** You can modify the code to train only the components that you care about.

- The `--exp` specifies the experiment name. `model_ref` as the experiment name is reserved for the respository of all the pre-trained models. You could run this script with a different experiment name, but then you would have to move all the trained models to `workspace/experiments/model_ref` to make it accessible to the entire code. This is useful if you want to have multiple sets of reference trained models -- it would make sense to run this script with different experiment names. You can also download all the reference models from [here](pretrained.md) and skip this step.
- The `--save` specifies that you'd like to save the pre-trained models on disk. If you just want to monitor or test the training procedure, you can run this script without this flag.

## The Mechanics of Training Reference Models
There are multiple types of reference model training.

1. Classification training with `NLLoss` (or `CE Loss` in the paper). This is the basic way of training prediction functions with k-classes.
2. K-way logistic regression training with `BCELoss`. In this setting, instead of cross-entropy loss, which assumes mutual exclusion, we use the binary cross-entropy loss which does not impose a mutual exclusion assumption. For making a prediction, we take the maximum of the independent activations.
3. Deep Ensemble training [1]. This is a variation of basic classification training, except the loss function is augmented with an adversarial loss term and we train 5 independent networks as an ensemble. For each sample, we generate an adversarial sample using the fast gradient-sign method, and augment the training batch with the adversarial examples.
4. (Variational)-Autoencoder training with `BCE` loss or `MSE` loss.
5. Pixel-CNN++ training.

Each type has a list of _datasets_ and the corresponding _networks_ to be trained with a _training function_ to be called (see `task_list` in [model_setup.py](../setup/model_setup.py)). You can follow the training functions to see how each model is trained. Individual implementations are in [setup/categories/](../setup/categories/).

For the classification task, we instantiate datasets and network architectures according to the information in [global_vars.py](../global_vars.py). See Global Variables in [code organization](code_organization.md#global-variables) for more information.

Note that these networks are only trained with `D1_train` set. We further randomly split `D1_train` into [0.8, 0.2] sets and only train on the 0.8 portion of the data while using the remainder 0.2 as the test set. Therefore, the performance of the trained networks cannot be directly compared with the state-of-the-art performance in the respective datasets as the measurements and the training sets are different.

For data augmentation, we explicitly instantiate mirror augmented data instead of random on-air augmentation. *We do not allow any other augmentation* to ensure fair comparison in our paper. But if you would rather augment the hell of out of the data you should (i) explicitly document the type of data augmentation, and (ii) be extremely careful with fair comparison on other methods.

# References
1. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In NIPS.
