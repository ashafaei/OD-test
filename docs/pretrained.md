# Downloading the Pretrained Models

The pretrained models must be located in `workspace/experiments/model_ref`.

You can download all the models (13 GB uncompressed) at once, or download only the models you need.

- Download the entire set of reference models from [http://cs.ubc.ca/~shafaei/dataset/odtest/model_ref.tar.gz](http://www.cs.ubc.ca/~shafaei/dataset/odtest/model_ref.tar.gz).
- Navigate through the models [http://cs.ubc.ca/~shafaei/dataset/odtest/](http://www.cs.ubc.ca/~shafaei/dataset/odtest/).

## If you download the full archive
If you put the compressed file in the `<root_folder>` and extract with

```bash
> cd <root_folder> # so that workspace/experiments/model_ref would be immediately visible.
> tar xzfv model_ref.tar.gz
```

it will automatically extract the files to `<root_folder>/workspace/experiments/model_ref` (the archive is made from inside the `<root_folder>`.)

## If you want to download manually
By visiting [http://www.cs.ubc.ca/~shafaei/dataset/odtest/](http://www.cs.ubc.ca/~shafaei/dataset/odtest/) you get a directory listing of all the relevant files. You can navigate and download the models that you like. Make sure that you preserve the directory names when downloading the files.

The easy way to fetch a subset of the pretrained models is to use `wget` with the recursive option like this example.
```bash
> cd <root_folder> # so that workspace/experiments/model_ref would be immediately visible.
> wget -r -nH -np --cut-dirs=3 --reject="index.html*" -e robots=off \
    http://www.cs.ubc.ca/~shafaei/dataset/odtest/workspace/experiments/model_ref/MNIST_VGG.HClass/
```
You can run this with your own preferred subdirectory. All the files below the address would be downloaded and correctly put in the right directory of the project.

# Pre-trained Reference Models

The classification performance of the shared models on the entire `D1_train` is as follows:

|Dataset        | VGG               |  Resnet           |
|--------       |-----              |--------           |
| MNIST         | 99.89% (19 MB)    | 99.89% (70 MB)    |
| FashionMNIST  | 98.82% (19 MB)    | 98.75% (70 MB)    |
| CIFAR10       | 97.63% (159.8 MB) | 97.75% (94.3 MB)  |
| CIFAR100      | 91.40% (161.3 MB) | 92.05% (95.1 MB)  |
| TinyImagenet  | 69.71% (162.9 MB) | 89.59% (95.9 MB)  |
| STL10         | 93.62% (201.7 MB) | 92.32% (94.3 MB)  |

The `KWayLogistic` classification performance of the shared models on the entire `D1_train` is as follows:

|Dataset        | VGG               |  Resnet           |
|--------       |-----              |--------           |
| MNIST         | 99.91% (19 MB)    | 99.91% (70 MB)    |
| FashionMNIST  | 98.36% (19 MB)    | 98.73% (70 MB)    |
| CIFAR10       | 97.34% (159.8 MB) | 97.51% (94.3 MB)  |
| CIFAR100      | 91.87% (161.3 MB) | 91.82% (95.1 MB)  |
| TinyImagenet  | 72.23% (162.9 MB) | 65.95% (95.9 MB)  |
| STL10         | 95.18% (201.7 MB) | 93.34% (94.3 MB)  |

