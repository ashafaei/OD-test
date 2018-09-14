from __future__ import print_function
from os import path
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger

from methods import AbstractMethodInterface, AbstractModelWrapper, SVMLoss
from methods.base_threshold import ProbabilityThreshold
from datasets import MirroredDataset
import global_vars as Global

class RTModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        decision function tau > (x-mu)^2
    """
    def __init__(self, base_model, loss_variant=0):
        super(RTModelWrapper, self).__init__(base_model)
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0.5])))
        self.H.register_parameter('transfer', nn.Parameter(torch.FloatTensor([0.0])))
        self.loss_variant = loss_variant
        if self.loss_variant == 0:
            print(colored('BCE Loss', 'green'))
        else:
            print(colored('MSE Loss', 'green'))
        # from visdom import Visdom
        # self.visdom = Visdom(ipv6=False)

    def calculate_loss(self, input, target):
        loss = None
        if self.loss_variant == 0:
            loss = Fn.binary_cross_entropy_with_logits(input, target, size_average=False, reduce=False)
        else:
            loss = Fn.mse_loss(input, target, size_average=False, reduce=False)

        return loss.detach()

    def subnetwork_eval(self, x):
        x.requires_grad = False
        base_output = self.base_model(x).detach()
        loss = self.calculate_loss(base_output, x)
        loss = loss.view(loss.size(0), -1).mean(dim=1, keepdim=True)
        # self.visdom.images(x.data.cpu().numpy(), win='input')
        # self.visdom.images(nn.functional.sigmoid(base_output).data.cpu().numpy(), win='output')
        return loss

    def wrapper_eval(self, x):
        x = x - self.H.transfer
        x = x * x
        output = x - self.H.threshold
        return output
    
    def classify(self, x):
        return (x>0).long()

class ReconstructionThreshold(ProbabilityThreshold):
    def method_identifier(self):
        output = "REThreshold"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output        
    
    def get_base_config(self, dataset):
        print("Preparing training D1 for %s"%(dataset.name))

        # Initialize the multi-threaded loaders.
        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        # Set up the model
        model = Global.get_ref_autoencoder(dataset.name)[0]().to(self.args.device)

        # Set up the criterion
        criterion = None
        if self.default_model == 0:
            criterion = nn.BCEWithLogitsLoss().to(self.args.device)
        else:
            criterion = nn.MSELoss().to(self.args.device)
            model.default_sigmoid = True

        # Set up the config
        config = IterativeTrainerConfig()

        config.name = '%s-AE1'%(self.args.D1)
        config.phases = {
                        'all':     {'dataset' : all_loader,    'backward': False},
                        }
        config.criterion = criterion
        config.classification = False
        config.cast_float_label = False
        config.autoencoder_target = True
        config.stochastic_gradient = True
        config.visualize = not self.args.no_visualize
        config.sigmoid_viz = self.default_model == 0
        config.model = model
        config.optim = None
        config.logger = Logger()

        return config

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        import models as Models
        if self.default_model == 0:
            config.model.netid = "BCE." + config.model.netid
        else:
            config.model.netid = "MSE." + config.model.netid

        home_path = Models.get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name, suffix_str=config.model.netid)
        hbest_path = path.join(home_path, 'model.best.pth')
        best_h_path = hbest_path

        trainer = IterativeTrainer(config, self.args)

        if not path.isfile(best_h_path):
            raise NotImplementedError("%s not found!, Please use setup_model to pretrain the networks first!"%best_h_path)
        else:
            print(colored('Loading H1 model from %s'%best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))
        
        trainer.run_epoch(0, phase='all')
        test_loss = config.logger.get_measure('all_loss').mean_epoch(epoch=0)
        print("All average loss %s"%colored('%.4f'%(test_loss), 'red'))

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

        # Set up the criterion
        # To make the threshold learning, actually threshold learning
        # the margin must be set to 0.
        criterion = SVMLoss(margin=0.0).to(self.args.device)

        # Set up the model
        model = RTModelWrapper(self.base_model, loss_variant=self.default_model).to(self.args.device)

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
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
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-1, weight_decay=0)
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100

        return config
