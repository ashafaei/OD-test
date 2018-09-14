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

The second column is the method and the config. `prob_threshold/0` means the probability threshold method was run with the config `0` which is the VGG architecture (the index of architecture in the `global_vars`). The third column is `D_s - D_v` and the fourth column is `train_error / test_error` for the method. We only use the `test_error` for figures and anlysis.

If you have made it this far, it means everything is set up and ready to use. Next we explain what is going on in the script.

## Evaluation
