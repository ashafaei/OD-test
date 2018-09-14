from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
import os
from os import path
from termcolor import colored

from methods.base_threshold import ProbabilityThreshold
from datasets import MirroredDataset

class BinaryModelWrapper(nn.Module):
    """ The wrapper class for H.
        We add a layer at the end of any classifier. This module takes the |y| dimensional output
        and maps it to a one-dimensional prediction.
    """
    def __init__(self, base_model):
        super(BinaryModelWrapper, self).__init__()
        self.base_model = base_model
        output_size = base_model.output_size()[1].item()
        self.H = nn.Sequential(
                    nn.BatchNorm1d(output_size),
                    nn.Linear(output_size, 1),
        )

    def forward(self, x):
        base_output = self.base_model.forward(x, softmax=False)
        output = self.H(base_output)
        return output
    
    def preferred_name(self):
        return self.base_model.__class__.__name__
    
    def classify(self, x):
        return (x>0).long()

class BinaryClassifier(ProbabilityThreshold):
    def method_identifier(self):
        output = "BinaryClassifier"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_H_config(self, dataset, will_train=True):
        print("Preparing training D1+D2 (H)")
        print("Mixture size: %s"%colored('%d'%len(dataset), 'green'))
        import global_vars as Global

        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)

        if self.args.D1 in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%self.args.D1, 'green'))
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)
        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        criterion = nn.BCEWithLogitsLoss().cuda()

        # Set up the model
        model = Global.get_ref_classifier(self.args.D1)[self.default_model]().to(self.args.device)
        self.add_identifier = model.__class__.__name__
        if hasattr(model, 'preferred_name'):
            self.add_identifier = model.preferred_name()
        model = BinaryModelWrapper(model).to(self.args.device)

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = model.__class__.__name__
        if hasattr(model, 'preferred_name'):
            base_model_name = model.preferred_name()

        config.name = '_%s[%s](%s->%s)'%(self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = train_loader
        config.valid_loader = valid_loader
        config.phases = {
                        'train':   {'dataset' : train_loader,  'backward': True},
                        'test':    {'dataset' : valid_loader,  'backward': False},
                        'testU':   {'dataset' : all_loader, 'backward': False},                        
                        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = True
        config.stochastic_gradient = True
        config.visualize = not self.args.no_visualize
        config.model = model
        config.logger = Logger()
        config.optim = optim.Adam(model.parameters(), lr=1e-3)
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=5, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
        config.max_epoch = 30
        
        if hasattr(model, 'train_config'):
            model_train_config = model.train_config()
            for key, value in model_train_config.iteritems():
                print('Overriding config.%s'%key)
                config.__setattr__(key, value)

        return config

    def propose_H(self, dataset):
        raise NotImplementedError("You know, you're not supposed to treat me like this!")

    def train_H(self, dataset):
        # Wrap the (mixture)dataset in SubDataset so to easily
        # split it later. God knows how many wrappers we have by this point.
        from datasets import SubDataset
        dataset = SubDataset('%s-%s'%(self.args.D1, self.args.D2), dataset, torch.arange(len(dataset)).int())

        h_path = path.join(self.args.experiment_path, '%s'%(self.__class__.__name__),
                                                      '%d'%(self.default_model),
                                                      '%s->%s.pth'%(self.args.D1, self.args.D2))
        h_parent = path.dirname(h_path)
        if not path.isdir(h_parent):
            os.makedirs(h_parent)

        done_path = h_path + '.done'
        will_train = self.args.force_train_h or not path.isfile(done_path)

        h_config = self.get_H_config(dataset)

        trainer = IterativeTrainer(h_config, self.args)

        if will_train:
            print(colored('Training from scratch', 'green'))
            best_accuracy = -1
            trainer.run_epoch(0, phase='test')
            for epoch in range(1, h_config.max_epoch):
                trainer.run_epoch(epoch, phase='train')
                trainer.run_epoch(epoch, phase='test')

                train_loss = h_config.logger.get_measure('train_loss').mean_epoch()
                h_config.scheduler.step(train_loss)

                # Track the learning rates and threshold.
                lrs = [float(param_group['lr']) for param_group in h_config.optim.param_groups]
                h_config.logger.log('LRs', lrs, epoch)
                h_config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
                if h_config.visualize:
                    # Show the average losses for all the phases in one figure.
                    h_config.logger.visualize_average_keys('.*_loss', 'Average Loss', trainer.visdom)
                    h_config.logger.visualize_average_keys('.*_accuracy', 'Average Accuracy', trainer.visdom)
                    h_config.logger.visualize_average('LRs', trainer.visdom)

                test_average_acc = h_config.logger.get_measure('test_accuracy').mean_epoch()

                # Save the logger for future reference.
                torch.save(h_config.logger.measures, path.join(h_parent, 'logger.%s->%s.pth'%(self.args.D1, self.args.D2)))

                if best_accuracy < test_average_acc:
                    print('Updating the on file model with %s'%(colored('%.4f'%test_average_acc, 'red')))
                    best_accuracy = test_average_acc
                    torch.save(h_config.model.state_dict(), h_path)

                if test_average_acc > 1-1e-4:
                    break

            torch.save({'finished':True}, done_path)

            if h_config.visualize:
                trainer.visdom.save([trainer.visdom.env])

        # Load the best model.
        print(colored('Loading H model from %s'%h_path, 'red'))
        h_config.model.load_state_dict(torch.load(h_path))
        
        trainer.run_epoch(0, phase='testU')
        test_average_acc = h_config.logger.get_measure('testU_accuracy').mean_epoch(epoch=0)
        print("Valid/Test average accuracy %s"%colored('%.4f%%'%(test_average_acc*100), 'red'))
        self.H_class = h_config.model
        self.H_class.eval()
        return test_average_acc
