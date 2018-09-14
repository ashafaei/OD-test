from __future__ import print_function
from os import path
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim

from torch import autograd
from methods import AbstractModelWrapper, SVMLoss
from methods.base_threshold import ProbabilityThreshold
from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
import global_vars as Global
from datasets import MirroredDataset

class DeepEnsembleWrapper(nn.Module):
    def __init__(self, parent_model):
        super(DeepEnsembleWrapper, self).__init__()
        self.model = parent_model
        # We need to keep track of the previous X
        # to efficiently calculate the perturbed loss.
        self.previous_X = None
    
    def forward(self, x, **kwargs):
        if self.training:
            x.requires_grad = True
            self.previous_X = x
            if x.grad is not None:
                x.grad.zero_()
        model_output = self.model(x, **kwargs)
        return model_output

    def preferred_name(self):
        return self.model.__class__.__name__

    def output_size(self):
        return self.model.output_size()
        

class DeepEnsembleLoss(nn.Module):
    def __init__(self, ensemble_network, epsilon=0.01):
        super(DeepEnsembleLoss, self).__init__()
        assert ensemble_network.__class__ == DeepEnsembleWrapper, 'Only EnsembleWrappers are accepted.'
        self.ensemble_network = ensemble_network
        self.epsilon = epsilon
        self.size_average = True
        self.loss = nn.NLLLoss(size_average=self.size_average)
        
    def forward(self, input, target):
        """
            In deep ensembles, we optimize the following objective:
            l(w, X, Y) + l(w, X', Y) where X' is the X FGSM-perturbed sample.
        """
        # Let's calculate the first part of the loss.
        loss_1 = self.loss(input, target)

        total_loss = loss_1

        # During test, we don't do this.
        if self.ensemble_network.training:
            # Let's do the backward pass.
            input_x = self.ensemble_network.previous_X
            grad_input_x = autograd.grad([loss_1], [input_x], retain_graph=True, only_inputs=True)[0]

            # construct X' - Fast Gradient Sign Method + Projection
            new_input = (input_x.detach() + 0.01*grad_input_x.detach().sign()).clamp(min=0, max=1)
            new_input.detach_()
            new_input.requires_grad=False
            new_output = self.ensemble_network.model(new_input)

            # Calculate the second term.
            loss_2 = self.loss(new_output, target)

            total_loss = loss_1 + loss_2

        return total_loss        

class DeepEnsembleMasterWrapper(nn.Module):
    """
        This master wrapper evalutes and averages over multiple networks.
        Nothing special happenning here.
    """
    def __init__(self, subwrappers):
        assert subwrappers is not None
        super(DeepEnsembleMasterWrapper, self).__init__()
        self.subwrappers = subwrappers

    def forward(self, x, take_log=True, **kwargs):
        outputs = []
        for model in self.subwrappers:
            model.eval()
            predictions = model(x).unsqueeze(0).detach().exp() # Must average over the probabilities.
            outputs.append(predictions)
        output = torch.cat(outputs).mean(dim=0)
        if take_log:
            return output.log() # take the log for consistency with other models after averaging.
        else:
            return output
    def preferred_name(self):
        return self.subwrappers[0].preferred_name()

class DeepEnsembleModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
        This is the network that is actually saved on evaluations.
        We don't want to replicate multiple networks over each snapshot.
    """
    def __init__(self, base_model):
        super(DeepEnsembleModelWrapper, self).__init__(base_model)
        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0]))) # initialize to 0 for faster convergence.

    def subnetwork_eval(self, x):
        x.requires_grad = False
        average  = self.base_model(x, take_log=False).detach()
        output_tensor = (average * average.log()).sum(dim=1, keepdim=True)
        return output_tensor

    def wrapper_eval(self, x):
        # Threshold hold the uncertainty.
        output = self.H.threshold - x
        return output

    def classify(self, x):
        return (x>0).long()


class DeepEnsemble(ProbabilityThreshold):
    def method_identifier(self):
        output = "DeepEnsemble"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_base_config(self, dataset):
        print("Preparing training D1 for %s"%(dataset.parent_dataset.__class__.__name__))

        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        criterion = nn.NLLLoss().cuda()

        # Set up the model
        model_class = Global.get_ref_classifier(dataset.name)[self.default_model]
        self.add_identifier = model_class.__name__

        # We must create 5 instances of this class.
        from models import get_ref_model_path
        all_models = []
        for mid in range(5):
            model = model_class()
            model = DeepEnsembleWrapper(model)
            model = model.to(self.args.device)
            h_path = get_ref_model_path(self.args, model_class.__name__, dataset.name, suffix_str='DE.%d'%mid)
            best_h_path = path.join(h_path, 'model.best.pth')
            if not path.isfile(best_h_path):      
                raise NotImplementedError("Please use setup_model to pretrain the networks first! Can't find %s"%best_h_path)
            else:
                print(colored('Loading H1 model from %s'%best_h_path, 'red'))
                model.load_state_dict(torch.load(best_h_path))
                model.eval()
            all_models.append(model)
        master_model = DeepEnsembleMasterWrapper(all_models)

        # Set up the config
        config = IterativeTrainerConfig()

        config.name = '%s-CLS'%(self.args.D1)
        config.phases = {
                        'all':     {'dataset' : all_loader,    'backward': False},                        
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = False
        config.stochastic_gradient = True
        config.model = master_model
        config.optim = None
        config.autoencoder_target = False
        config.visualize = False
        config.logger = Logger()
        return config

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        """ This is really time consuming, especially for Resnet models.
            Feel free to uncomment these lines if you want to see the performance.
        """
        # trainer = IterativeTrainer(config, self.args)
        # trainer.run_epoch(0, phase='all')
        # test_average_acc = config.logger.get_measure('all_accuracy').mean_epoch(epoch=0)
        # print("All average accuracy %s"%colored('%.4f%%'%(test_average_acc*100), 'red'))

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
        model = DeepEnsembleModelWrapper(self.base_model).to(self.args.device)

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
