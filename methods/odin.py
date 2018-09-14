"""
    For ODIN[1], we simply apply
        f(x) = [max prob_t(x')]> theta, where 
        - x' is the epsilon-perturbed x in direction that minimizes the CE loss on **the most likely output class**.
            - For the loss calculation, the output is scaled according to the temperature.
                we use the same procedure as the reference code for the paper
                See https://github.com/ShiyuLiang/odin-pytorch/blob/34e53f5a982811a0d74baba049538d34efc0732d/code/calData.py#L48
        - prob_t is the softmax with temperature t. The max is over all labels.
        - theta is the threshold.
        We learn the 
        - threshold with an SVM loss (margin 0).
        - grid-search over the epsilon and temperature.

[1] S. Liang, Y. Li, and R. Srikant, "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks" ICLR, 2018
"""

from __future__ import print_function
import os
import os.path as path
import timeit
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader

import global_vars as Global
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
from methods.base_threshold import ProbabilityThreshold
from methods import AbstractModelWrapper, SVMLoss
from datasets import SubDataset, MirroredDataset

class ODINModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
    """
    def __init__(self, base_model, epsilon=0.0012, temperature=1000):
        super(ODINModelWrapper, self).__init__(base_model)
        # Let's have these fixed for now!

        self.H = nn.Module()
        self.H.register_parameter('threshold', nn.Parameter(torch.Tensor([0.5]))) # initialize to prob=0.5 for faster convergence.

        # register params under H for storage.
        self.H.register_buffer('epsilon', torch.FloatTensor([epsilon]))
        self.H.register_buffer('temperature', torch.FloatTensor([1./temperature]))

        self.criterion = nn.CrossEntropyLoss()

    def subnetwork_eval(self, x):
        # We have to backpropagate through the input.
        # The model must be fixed in the eval mode.
        new_x = x.clone()
        cur_x = x.clone()

        grad_input_x = None
        with torch.set_grad_enabled(True):
            cur_x.requires_grad = True
            if cur_x.grad is not None:
                cur_x.grad.zero_()
            base_output = self.base_model(cur_x, softmax=False)
            y_hat = base_output.max(1)[1].detach()
            base_output = base_output * self.H.temperature
            loss = self.criterion(base_output, y_hat)
            grad_input_x = autograd.grad([loss], [cur_x], retain_graph=False, only_inputs=True)[0]        

        # This code is written based on author's code.
        # https://github.com/ShiyuLiang/odin-pytorch/blob/34e53f5a982811a0d74baba049538d34efc0732d/code/calData.py#L183
        # construct X' - Fast Gradient Sign Method + Projection
        # They scale the gradient by 4.1 because of the normalization they apply
        # to the images. But we do a late normalization of images in the architecture
        # itself, so the gradient is scaled properly already. Though it doesn't really
        # matter, because we learn the Epsilon anyway. Normally you should project the 
        # perturbed image back to the hypercupe, but they don't do it. So I didn't either.
        new_input = (new_x.detach() - self.H.epsilon*(grad_input_x.detach().sign()))
        new_input.detach_()
        new_input.requires_grad = False

        # second evaluation.
        new_output = self.base_model(new_input, softmax=False).detach()

        new_output.mul_(self.H.temperature)

        probabilities = F.softmax(new_output, dim=1)

        # Get the max probability out
        input = probabilities.max(1)[0].detach().unsqueeze_(1)

        return input.detach()

    def wrapper_eval(self, x):
        # Threshold hold the max probability.
        output = self.H.threshold - x
        return output
    
    def classify(self, x):
        return (x>0).long()

class ODIN(ProbabilityThreshold):
    def method_identifier(self):
        output = "ODIN"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output        
    
    def get_H_config(self, train_ds, valid_ds, will_train=True, epsilon=0.0012, temperature=1000):
        print("Preparing training D1+D2 (H)")

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        # To make the threshold learning, actually threshold learning
        # the margin must be set to 0.
        criterion = SVMLoss(margin=0.0).to(self.args.device)

        # Set up the model
        model = ODINModelWrapper(self.base_model, epsilon=epsilon, temperature=temperature).to(self.args.device)

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
            x_center = trainX[trainY==0].mean()
            y_center = trainX[trainY==1].mean()
            init_value = (x_center+y_center)/2
            model.H.threshold.fill_(init_value)
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
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-2, weight_decay=0)
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=5, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 30

        return config

    def train_H(self, dataset):
        # Wrap the (mixture)dataset in SubDataset so to easily
        # split it later.
        dataset = SubDataset('%s-%s'%(self.args.D1, self.args.D2), dataset, torch.arange(len(dataset)).int())
        
        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)
        
        if self.args.D1 in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%self.args.D1, 'green'))
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # As suggested by the authors.
        all_temperatures = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        all_epsilons     = torch.linspace(0, 0.004, 21)
        total_params = len(all_temperatures) * len(all_epsilons)
        best_accuracy = -1

        h_path = path.join(self.args.experiment_path, '%s'%(self.__class__.__name__),
                                                      '%d'%(self.default_model),
                                                      '%s->%s.pth'%(self.args.D1, self.args.D2))
        h_parent = path.dirname(h_path)
        if not path.isdir(h_parent):
            os.makedirs(h_parent)

        done_path = h_path + '.done'
        trainer, h_config = None, None

        if self.args.force_train_h or not path.isfile(done_path):
            # Grid search over the temperature and the epsilons.
            for i_eps, eps in enumerate(all_epsilons):
                for i_temp, temp in enumerate(all_temperatures):
                    so_far = i_eps * len(all_temperatures) + i_temp + 1
                    print(colored('Checking eps=%.2e temp=%.1f (%d/%d)'%(eps, temp, so_far, total_params), 'green'))
                    start_time = timeit.default_timer()

                    h_config = self.get_H_config(train_ds=train_ds, valid_ds=valid_ds,
                                                epsilon=eps, temperature=temp)

                    trainer = IterativeTrainer(h_config, self.args)

                    print(colored('Training from scratch', 'green'))
                    trainer.run_epoch(0, phase='test')
                    for epoch in range(1, h_config.max_epoch+1):
                        trainer.run_epoch(epoch, phase='train')
                        trainer.run_epoch(epoch, phase='test')

                        train_loss = h_config.logger.get_measure('train_loss').mean_epoch()
                        h_config.scheduler.step(train_loss)

                        # Track the learning rates and threshold.
                        lrs = [float(param_group['lr']) for param_group in h_config.optim.param_groups]
                        h_config.logger.log('LRs', lrs, epoch)
                        h_config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]

                        if hasattr(h_config.model, 'H') and hasattr(h_config.model.H, 'threshold'):
                            h_config.logger.log('threshold', h_config.model.H.threshold.cpu().numpy(), epoch-1)
                            h_config.logger.get_measure('threshold').legend = ['threshold']
                            if h_config.visualize:
                                h_config.logger.get_measure('threshold').visualize_all_epochs(trainer.visdom)

                        if h_config.visualize:
                            # Show the average losses for all the phases in one figure.
                            h_config.logger.visualize_average_keys('.*_loss', 'Average Loss', trainer.visdom)
                            h_config.logger.visualize_average_keys('.*_accuracy', 'Average Accuracy', trainer.visdom)
                            h_config.logger.visualize_average('LRs', trainer.visdom)

                        test_average_acc = h_config.logger.get_measure('test_accuracy').mean_epoch()

                        if best_accuracy < test_average_acc:
                            print('Updating the on file model with %s'%(colored('%.4f'%test_average_acc, 'red')))
                            best_accuracy = test_average_acc
                            torch.save(h_config.model.H.state_dict(), h_path)

                    elapsed = timeit.default_timer() - start_time
                    print('Hyper-param check (%.2e, %.1f) in %.2fs' %(eps, temp, elapsed))

            torch.save({'finished':True}, done_path)

        # If we load the pretrained model directly, we will have to initialize these.
        if trainer is None or h_config is None:
            h_config = self.get_H_config(train_ds=train_ds, valid_ds=valid_ds,
                                        epsilon=0, temperature=1, will_train=False)
            # don't worry about the values of epsilon or temperature. it will be overwritten.
            trainer = IterativeTrainer(h_config, self.args)

        # Load the best model.
        print(colored('Loading H model from %s'%h_path, 'red'))
        h_config.model.H.load_state_dict(torch.load(h_path))
        h_config.model.set_eval_direct(False)        
        print('Temperature %s Epsilon %s'%(colored(h_config.model.H.temperature.item(),'red'), colored(h_config.model.H.epsilon.item(), 'red')))

        trainer.run_epoch(0, phase='testU')
        test_average_acc = h_config.logger.get_measure('testU_accuracy').mean_epoch(epoch=0)
        print("Valid/Test average accuracy %s"%colored('%.4f%%'%(test_average_acc*100), 'red'))
        self.H_class = h_config.model
        self.H_class.eval()
        self.H_class.set_eval_direct(False)        
        return test_average_acc
