# Introduction

The script [eval3d.py](../eval3d.py) ties everything in this project together. But before we go through how this script works, let us go through a quick example to make sure you are all good to go.

## A quick evaluation test of PbThreshold
In this quick example, we will evaluate the method *PbThreshold* + *VGG* on *MNIST* with *Uniform Noise* and *Normal Noise*.
Let's first navigate to the project and activate the environment. If you have not set up the project, please follow the setup instructions in [readme](../README.md#setup) first.

```bash
> cd <root_folder>
> source workspace/env/bin/activate
```

Now the environment is active. Next, we will download the pretrained VGG model for MNIST. You can read more about the pretrained models [here](pretrained.md). If you have already downloaded the pretrained models or have previously trained the model, you can just skip this step.

```bash
(env)> wget -r -nH -np --cut-dirs=3 --reject="index.html*" -e robots=off \
    http://www.cs.ubc.ca/~shafaei/dataset/odtest/workspace/experiments/model_ref/MNIST_VGG.HClass/MNIST.dataset/base/
```

Now you can run the evaluation code.

```bash
(env)> python eval3d.py --exp test-eval --no-visualize
```

The program should terminate with the following output:
```bash
0	prob_threshold/0	MNIST-UniformNoise	NormalNoise    	97.67% / 98.74%
1	prob_threshold/0	MNIST-NormalNoise	UniformNoise   	98.55% / 94.99%
```

The second column is the method and the config. `prob_threshold/0` means the probability threshold method was run with the config `0` which is the VGG architecture (the index of architecture in the `global_vars`). The third column is `D_s - D_v` and the fourth column is `D_t`. The fifth column is `train_error / test_error` for the method. We only use the `test_error` for figures and anlysis.

These results are also saved under `workspace/experiments/test-eval/results.pth` as a simple table that you can use to generate figures.

If you have made it this far, it means everything is set up and ready to use. Next we explain what is going on in the script.

## Evaluation

The evaluation script is simply a for-loop over all combinations of *datasets* and *methods*. At the time of this writing, the master configuration, which includes all the datasets and methods is as follows:
```python
d1_tasks     = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
method_tasks = [
                'pixelcnn/0',
                'mcdropout/0',
                'prob_threshold/0',     'prob_threshold/1',
                'score_svm/0',          'score_svm/1',
                'logistic_svm/0',       'logistic_svm/1',
                'openmax/0',            'openmax/1',
                'binclass/0',           'binclass/1',
                'deep_ensemble/0',      'deep_ensemble/1',
                'odin/0',               'odin/1',
                'reconst_thresh/0',     'reconst_thresh/1',
                'knn/1', 'knn/2', 'knn/4', 'knn/8',
                'bceaeknn/1', 'vaeaeknn/1', 'mseaeknn/1',
                'bceaeknn/2', 'vaeaeknn/2', 'mseaeknn/2',
                'bceaeknn/4', 'vaeaeknn/4', 'mseaeknn/4',
                'bceaeknn/8', 'vaeaeknn/8', 'mseaeknn/8',
                ]
```

You can run your own subset of evaluations by simply commenting out the unnecessary evaluations. The majority of evaluation logic is as follows:

```python
for d1 in d1_tasks:
    for d2 in d2_tasks:
        for d3 in d3_tasks:
            for method in method_tasks:
                # Algorithm 1 - Line 3 (in the paper)
                d1_train = d1.get_D1_train()
                method.propose_H(d1_train)

                # Algorithm 1 - Line 5
                d1_valid = d1.get_D1_valid()
                d2_valid = d2.get_D2_valid(d1)
                valid_mixture = d1_valid + d2_valid
                train_acc = method.train_H(valid_mixture)

                # Algorithm 1 - Line 7
                d1_test = d1.get_D1_test()
                d2_test = d3.get_D2_test(d1)
                test_mixture = d1_test + d2_test
                test_acc = method.test_H(test_mixture)
```

The actual implementation has more bells and whistles for efficiency and various other reasons. For `BinClass` which is a direct binary classification of `D_s` vs `D_v` the logic is slightly different.

As you can see, the datasets and the methods must implement certain functions for this evaluation to work correctly. See [code organization](code_organization.md) to learn more.

The evaluation code saves a snapshot of the results after each evaluation. You can load the results from `workspace/experiments/<exp-name>/results.pth`. If you run the same experiment, the script would automatically skip over the experiments that are already done. The intermediate reject functions that are learned during the evaluations are also saved under `workspace/experiments/<exp-name>/<method-name>/`.

When you run `eval3d.py` the following experiment names are pre-defined:
- `master`: Runs on every dataset and method.
- `eval-test`: Runs the quick demo from above.
everything else would go to the default config
```python
d1_tasks     = ['MNIST']
d2_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
d3_tasks     = ['UniformNoise', 'NormalNoise', 'MNIST', 'FashionMNIST', 'NotMNIST', 'CIFAR10', 'CIFAR100', 'STL10', 'TinyImagenet']
method_tasks     = [
                    'prob_threshold/0',
                    ]

```
which you can customize to your needs.