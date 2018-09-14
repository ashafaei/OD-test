from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from termcolor import colored

from methods import AbstractModelWrapper, SVMLoss
import global_vars as Global
from datasets import MirroredDataset
from methods.base_threshold import ProbabilityThreshold

class MCDropoutModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        For MCDropout we run the dropouts in train_mode.
    """
    def __init__(self, base_model):
        super(MCDropoutModelWrapper, self).__init__(base_model)
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0]))) # initialize to 0 for faster convergence.
        self.H.register_buffer('n_evals', torch.IntTensor([7]))

    def subnetwork_eval(self, x):
        # On MCDropout, we set the dropouts to train mode.
        count = 0
        for m in self.base_model.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train(True)
                count += 1
        assert count > 0, 'We can only do models with dropout!'

        x.requires_grad = False
        n_evals = self.H.n_evals.item()

        process_input = x.repeat(n_evals, 1, 1, 1)
        unprocessed_output = self.base_model(process_input).detach().exp()
        average  = unprocessed_output.view(n_evals, x.size(0), -1).mean(dim=0)

        output_tensor = (average * average.log()).sum(dim=1, keepdim=True)

        return output_tensor

    def wrapper_eval(self, x):
        # Threshold hold the uncertainty.
        output = self.H.threshold - x
        return output
    
    def classify(self, x):
        return (x>0).long()

class MCDropout(ProbabilityThreshold):
    def method_identifier(self):
        output = "MCDropout-7"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output
    
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
        model = MCDropoutModelWrapper(self.base_model).to(self.args.device)

        old_valid_loader = valid_loader

        # By definition, this approach is uncacheable :(

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
