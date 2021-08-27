import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models.vgg as VGG
import torchvision.models.resnet as Resnet

import torchinfo

class Scaled_VGG(nn.Module):

    #channel-aware VGG builder
    def make_layers(self, cfg, channels, batch_norm=False):
        layers = []
        in_channels = channels
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

    def __init__(self,scale,classes,epochs,init_weights=True):
        super(Scaled_VGG, self).__init__()

        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:0')

        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        small_cfg = [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        middle_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        large_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        classifier_width = 4096

        self.scale = scale

        if scale[1]<32:
           self.cfg = small_cfg
           classifier_width = 256

        if scale[1]==32:
            self.cfg = middle_cfg

        if scale[1]>32:
            self.cfg = large_cfg

        maxpool_count = self.cfg.count('M')
        scale_factor = 2**maxpool_count

        channels = scale[0]
        self.model = VGG.VGG(self.make_layers(self.cfg, channels, batch_norm=True), num_classes=classes)
        # would have a different sized feature map.
        poolscale = ((int)(scale[0]/scale_factor), (int)(scale[1]/scale_factor), (int)(scale[2]/scale_factor));
        self.model.avgpool = nn.AdaptiveAvgPool2d((poolscale[1],poolscale[2]))
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * poolscale[1] * poolscale[2], classifier_width), nn.ReLU(True), nn.Dropout(),
            nn.Linear(classifier_width, classifier_width), nn.ReLU(True), nn.Dropout(),
            nn.Linear(classifier_width, classes),
        )

        self.model = self.model.to(self.dev1)

        torchinfo.summary(self.model, col_names=["kernel_size", "input_size", "output_size", "num_params"], input_size=(32, self.scale[0], self.scale[1], self.scale[2]))

        if(init_weights):
            self.model._initialize_weights()
        
        self.epochs = epochs

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = x.to(self.dev1)
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)

        output = output.to(self.dev2)
        return output

    def get_info(self,args):
        self.batch_size = args.batch_size
        self.info = torchinfo.summary(self.model, input_size=(self.batch_size, self.scale[0], self.scale[1], self.scale[2]), verbose=0)
        return self.info
    
    def get_output_device(self):
        return torch.device('cuda:0')

    def output_size(self):
        return torch.LongTensor([1, classes])

    # because the model is split, we need to know which device the outputs go to put the labels on so the loss function can do the comparison
    def get_output_device(self):
        return self.dev2

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = self.epochs
        return config

class Scaled_VGG_2GPU(Scaled_VGG):

    def __init__(self,scale,classes,epochs):
        super(Scaled_VGG_2GPU, self).__init__(scale,classes,epochs,False)

        #self.dev1 = torch.device('cuda:0')
        #self.dev2 = torch.device('cpu')

        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:1')

        # features on GPU0, classifier on GPU1
        self.model.features.to(self.dev1)
        self.model.avgpool.to(self.dev1)
        self.seq1 = nn.Sequential(
            self.model.features,
            self.model.avgpool
            ).to(self.dev1)

        self.model.classifier.to(self.dev2)
        self.seq2 = nn.Sequential(
            self.model.classifier
            ).to(self.dev2)

        self._initialize_weights

    def forward(self, x, softmax=True):
        x = x.to(self.dev1)
        x = (x-self.offset)*self.multiplier
        x = self.seq1(x)
        x = torch.flatten(x, 1)
        x = x.to(self.dev2)
        output = self.seq2(x)

        if softmax:
            output = F.log_softmax(output, dim=1)
        return output
    
    def _initialize_weights(self) -> None:
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # because the model is split, we need to know which device the outputs go to put the labels on so the loss function can do the comparison
    def get_output_device(self):
        return self.dev2

    def output_size(self):
        return torch.LongTensor([1, classes])

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = self.epochs
        return config

class Scaled_VGG_2GPU_Pipeline(Scaled_VGG_2GPU):
    #taken pretty straight from https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
    def __init__(self, split_size=8, *args, **kwargs):
        super(Scaled_VGG_2GPU_Pipeline, self).__init__(*args,**kwargs)
        self.split_size = split_size
            
    def forward(self, x, softmax=True):
        splits = iter(x.split(self.split_size, dim=0))

        s_next = next(splits).to(self.dev1)
        s_next = (s_next-self.offset)*self.multiplier
        s_prev = self.seq1(s_next)
        s_prev = torch.flatten(s_prev, 1)
        s_prev = s_prev.to(self.dev2)

        ret = []

        for s_next in splits:
            s_next = s_next.to(self.dev1)
            s_next = (s_next-self.offset)*self.multiplier
            # A. s_prev runs on cuda:1
            output = self.seq2(s_prev)
            ret.append(output)

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next)
            s_prev = torch.flatten(s_prev, 1)
            s_prev = s_prev.to(self.dev2)

        output = self.seq2(s_prev)
        ret.append(output)

        p = torch.cat(ret)
        if softmax:
            p = F.log_softmax(p, dim=1)

        return p

class Scaled_Resnet(nn.Module):
    """
        MNIST_Resnet is based on Resnet50
        We replace the average pooling block to accomodate
        the requirements of MNIST.
    """
    def __init__(self,scale,classes,epochs):
        super(Scaled_Resnet, self).__init__()
        # Based on the imagenet normalization params.
        self.offset = 0.44900
        self.multiplier = 4.42477

        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:0')

        # Resnet50.
        layers = [2, 3, 5, 2]
        if scale[0] > 1:
            layers = [3, 4, 6, 3]
        self.model = Resnet.ResNet(Resnet.Bottleneck,layers, num_classes=classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # The first part also needs to be fixed.
        self.model.conv1 = nn.Conv2d(scale[0], 64, kernel_size=3, stride=1, padding=1, bias=False) # Replace the harsh convolution.
        del self.model.maxpool
        self.model.maxpool = lambda x: x # Remove the early maxpool.

        self.model = self.model.to(self.dev1)

        self.epochs = epochs

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = x.to(self.dev1)
        x = (x-self.offset)*self.multiplier

        output = self.model(x)
        if softmax:
            output = F.log_softmax(output, dim=1)

        output = output.to(self.dev2)
        return output

    def get_output_device(self):
        return torch.device('cuda:0')

    def output_size(self):
        return torch.LongTensor([1, classes])

    def get_output_device(self):
        return self.dev2

    def train_config(self):
        config = {}
        config['optim']     = optim.Adam(self.parameters(), lr=1e-3)
        config['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(config['optim'], patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config['max_epoch'] = self.epochs
        return config

class Scaled_Resnet_2GPU(Scaled_Resnet):

    def __init__(self,scale,classes,epochs):
        super(Scaled_Resnet_2GPU,self).__init__(scale,classes,epochs)
        
        self.dev1 = torch.device('cuda:0')
        self.dev2 = torch.device('cuda:1')

        self.seq1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,

            self.model.layer1,
            self.model.layer2
        ).to(self.dev1)

        self.seq2 = nn.Sequential(
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool,
        ).to(self.dev2)

        self.model.fc.to(self.dev2)    

    def forward(self, x, softmax=True):
        # Perform late normalization.
        x = (x-self.offset)*self.multiplier
        x.to(self.dev1)

        x = self.seq1(x).to(self.dev2)
        x = self.seq2(x)
        output = self.model.fc(x.view(x.size(0), -1))	

        if softmax:
            output = F.log_softmax(output, dim=1)
        return output

    def get_output_device(self):
        return self.dev2


class Scaled_Resnet_2GPU_Pipeline(Scaled_Resnet_2GPU):
    #taken pretty straight from https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
    def __init__(self, split_size=8, *args, **kwargs):
        super(Scaled_Resnet_2GPU_Pipeline, self).__init__(*args,**kwargs)
        self.split_size = split_size
            
    def forward(self, x, softmax=True):
        splits = iter(x.split(self.split_size, dim=0))

        s_next = next(splits)
        s_prev = self.seq1(s_next).to(self.dev2)
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            output = self.model.fc(s_prev.view(s_prev.size(0), -1))

            ret.append(output)

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to(self.dev2)

        s_prev = self.seq2(s_prev)
        output = self.model.fc(s_prev.view(s_prev.size(0), -1))
        ret.append(output)

        p = torch.cat(ret)

        if softmax:
            p = F.log_softmax(p, dim=1)

        return p
