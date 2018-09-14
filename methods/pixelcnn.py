from __future__ import print_function
from os import path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
from termcolor import colored

from methods import AbstractModelWrapper, SVMLoss
import global_vars as Global
from datasets import MirroredDataset
from methods.base_threshold import ProbabilityThreshold
import models.pixelcnn.model as PCNNModel
from models.pixelcnn.utils import PCNN_Loss

class PixelCNNModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        We get the log-likelihood from PixelCNN models.
    """
    def __init__(self, base_model):
        assert isinstance(base_model, PCNNModel.PixelCNN), 'We only support PixelCNN objects.'
        super(PixelCNNModelWrapper, self).__init__(base_model)
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0]))) # initialize to 0 for faster convergence.
        self.loss_func = PCNN_Loss(one_d = (base_model.input_channels==1))

    def subnetwork_eval(self, x):
        output_tensor = None
        self.base_model.eval()
        with torch.set_grad_enabled(False):
            x.requires_grad = False
            output = self.base_model(x)
            output_tensor = self.loss_func(output, x, do_reduce=False).data
        return output_tensor

    def wrapper_eval(self, x):
        # Threshold hold the NLL.
        # The lower log-likelihood has a higher probability.
        output = x - self.H.threshold
        return output
    
    def classify(self, x):
        return (x>0).long()

class PixelCNN(ProbabilityThreshold):
    def method_identifier(self):
        output = "PixelCNN"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output
    
    def get_base_config(self, dataset):
        print("Preparing training D1 for %s"%(dataset.parent_dataset.__class__.__name__))

        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        # Set up the model
        model = Global.get_ref_pixelcnn(dataset.name)[self.default_model]().to(self.args.device)
        self.add_identifier = model.__class__.__name__

        # Load the snapshot
        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, model.__class__.__name__, dataset.name, suffix_str=model.netid)
        best_h_path = path.join(h_path, 'model.best.pth')
        if not path.isfile(best_h_path):
            raise NotImplementedError("Please use setup_model to pretrain the networks first! Can't find %s"%best_h_path)
        else:
            print(colored('Loading H1 model from %s'%best_h_path, 'red'))
            model.load_state_dict(torch.load(best_h_path))
            model.eval()

        # Set up the criterion
        criterion = PCNN_Loss(one_d = (model.input_channels==1)).to(self.args.device)

        # Set up the config
        config = IterativeTrainerConfig()

        config.name = '%s-pcnn'%(self.args.D1)
        config.phases = {
                        'all':     {'dataset' : all_loader,    'backward': False},                        
                        }
        config.criterion = criterion
        config.classification = False
        config.cast_float_label = False
        config.autoencoder_target = True
        config.stochastic_gradient = True
        config.model = model
        config.optim = None
        config.visualize = False
        config.logger = Logger()
        return config

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        """ This is really time consuming.
            Feel free to uncomment these lines if you want to see the performance.
        """
        # trainer = IterativeTrainer(config, self.args)
        # trainer.run_epoch(0, phase='all')
        # test_average_loss = config.logger.get_measure('all_loss').mean_epoch(epoch=0)
        # print("All average loss (bpd)  %s"%colored('%.4f'%(test_average_loss), 'red'))

        self.base_model = config.model
        self.base_model.eval()

    def get_H_config(self, dataset, will_train=True):
        print("Preparing training D1+D2 (H)")
        print("Mixture size: %s"%colored('%d'%len(dataset), 'green'))

        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)

        if self.args.D1 in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%self.args.D1, 'green'))
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)

        # To make the threshold learning, actually threshold learning
        # the margin must be set to 0.
        criterion = SVMLoss(margin=0.0).to(self.args.device)

        # Set up the model
        model = PixelCNNModelWrapper(self.base_model).to(self.args.device)

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
            # Init the threshold.
            x_center = trainX[trainY==0].mean()
            y_center = trainX[trainY==1].mean()
            init_value = (x_center+y_center)/2
            model.H.threshold.data.fill_(init_value.item())
            print("Initializing threshold to %.2f"%(init_value.item()))

            new_valid_ds = TensorDataset(validX, validY)

            # Initialize the new multi-threaded loaders.
            train_loader = DataLoader(new_train_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)
            valid_loader = DataLoader(new_valid_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)

            # Set model to direct evaluation (for cached data)
            model.set_eval_direct(True)

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s->%s)'%(self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = train_loader
        config.valid_loader = valid_loader
        config.phases = {
                        'train':   {'dataset' : train_loader,  'backward': True},
                        'test':    {'dataset' : valid_loader,  'backward': False},
                        'testU':   {'dataset' : old_valid_loader, 'backward': False},                                                
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = True
        config.stochastic_gradient = True
        config.visualize = not self.args.no_visualize  
        config.model = model
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-1, weight_decay=1.0/len(train_ds))
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100

        return config
