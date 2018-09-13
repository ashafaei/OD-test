import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models.vgg as VGG
import torchvision.models.resnet as Resnet

class MNIST_VGG(nn.Module):
    """
        VGG-style MNIST.
    """

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def __init__(self):
        super(MNIST_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Reduced VGG16.
        self.cfg = [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.model = VGG.VGG(self.make_layers(self.cfg, batch_norm=True), num_classes=10)
        # MNIST would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 256), nn.ReLU(True), nn.Dropout(),
            nn.Linear(256, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output
    
    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config

class MNIST_Resnet(nn.Module):
    """
        MNIST_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of MNIST.
    """
    def __init__(self):
        super(MNIST_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [2, 3, 5, 2], num_classes=10)

        # MNIST would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config

class CIFAR10_VGG(nn.Module):
    """
        CIFAR_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR10_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # VGG16 minus last maxpool.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=10)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config
        
class CIFAR10_Resnet(nn.Module):
    """
        CIFAR_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR10_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)

        # Cifar 10 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 60
        return config
        
class CIFAR100_VGG(nn.Module):
    """
        CIFAR_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR100_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # VGG16 minus last maxpool.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=100)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 100),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 100])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config
        
class CIFAR100_Resnet(nn.Module):
    """
        CIFAR_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of CIFAR.
    """
    def __init__(self):
        super(CIFAR100_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=100)

        # Cifar 100 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 100])
        
    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class STL10_VGG(nn.Module):
    """
        STL10_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of STL10.
    """
    def __init__(self):
        super(STL10_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # VGG16.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=10)
        # Cifar 10 would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class STL10_Resnet(nn.Module):
    """
        STL10_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of STL10.
    """
    def __init__(self):
        super(STL10_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=10)

        # STL10 would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 10])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class TinyImagenet_VGG(nn.Module):
    """
        TinyImagenet_VGG is based on VGG16+BatchNorm
        We replace the classifier block to accomodate
        the requirements of TinyImagenet.
    """
    def __init__(self):
        super(TinyImagenet_VGG, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.model = VGG.VGG(VGG.make_layers(self.cfg, batch_norm=True), num_classes=200)
        # TinyImagenet would have a different sized feature map.
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 200),
        )
        self.model._initialize_weights()

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 200])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config

class TinyImagenet_Resnet(nn.Module):
    """
        TinyImagenet_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of TinyImagenet.
    """
    def __init__(self):
        super(TinyImagenet_Resnet, self).__init__()

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        # Resnet50.
        self.model = Resnet.ResNet(Resnet.Bottleneck, [3, 4, 6, 3], num_classes=200)

        # TinyImagenet would have a different sized feature map.
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        # del self.model.maxpool
        # self.model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def output_size(self):
        return torch.LongTensor([1, 200])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = 120
        return config
