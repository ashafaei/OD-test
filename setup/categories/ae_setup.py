from __future__ import print_function
import os
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models as Models
import global_vars as Global
from utils.iterative_trainer import IterativeTrainer, IterativeTrainerConfig
from utils.logger import Logger
from datasets import MirroredDataset

from models.autoencoders import VAE_Loss

def get_ae_config(args, model, dataset, BCE_Loss):
    print("Preparing training D1 for %s"%(dataset.name))

    # 80%, 20% for local train+test
    train_ds, valid_ds = dataset.split_dataset(0.8)

    if dataset.name in Global.mirror_augment:
        print(colored("Mirror augmenting %s"%dataset.name, 'green'))
        new_train_ds = train_ds + MirroredDataset(train_ds)
        train_ds = new_train_ds

    # Initialize the multi-threaded loaders.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    all_loader   = DataLoader(dataset,  batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # Set up the model
    model = model.to(args.device)

    # Set up the criterion
    criterion = None
    if BCE_Loss:
        criterion = nn.BCEWithLogitsLoss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)
        model.default_sigmoid = True

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'autoencoder_%s_%s'%(dataset.name, model.preferred_name())

    config.train_loader = train_loader
    config.valid_loader = valid_loader
    config.phases = {
                    'train':   {'dataset' : train_loader,  'backward': True},
                    'test':    {'dataset' : valid_loader,  'backward': False},
                    'all':     {'dataset' : all_loader,    'backward': False},                        
                    }
    config.criterion = criterion
    config.classification = False
    config.cast_float_label = False
    config.autoencoder_target = True
    config.stochastic_gradient = True
    config.visualize = not args.no_visualize
    config.sigmoid_viz = BCE_Loss
    config.model = model
    config.logger = Logger()

    config.optim = optim.Adam(model.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
    config.max_epoch = 120
    
    if hasattr(model, 'train_config'):
        model_train_config = model.train_config()
        for key, value in model_train_config.iteritems():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config

def get_vae_config(args, model, dataset):
    print("Preparing training D1 for %s"%(dataset.name))

    # 80%, 20% for local train+test
    train_ds, valid_ds = dataset.split_dataset(0.8)

    if dataset.name in Global.mirror_augment:
        print(colored("Mirror augmenting %s"%dataset.name, 'green'))
        new_train_ds = train_ds + MirroredDataset(train_ds)
        train_ds = new_train_ds

    # Initialize the multi-threaded loaders.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    all_loader   = DataLoader(dataset,  batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # Set up the model
    model = model.to(args.device)

    # Set up the criterion
    criterion = VAE_Loss(model)

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'vae_%s_%s'%(dataset.name, model.preferred_name())

    config.train_loader = train_loader
    config.valid_loader = valid_loader
    config.phases = {
                    'train':   {'dataset' : train_loader,  'backward': True},
                    'test':    {'dataset' : valid_loader,  'backward': False},
                    'all':     {'dataset' : all_loader,    'backward': False},                        
                    }
    config.criterion = criterion
    config.classification = False
    config.cast_float_label = False
    config.autoencoder_target = True
    config.stochastic_gradient = True
    config.visualize = not args.no_visualize
    config.sigmoid_viz = False
    config.model = model
    config.logger = Logger()

    config.optim = optim.Adam(model.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-3, min_lr=1e-6, factor=0.1, verbose=True)
    config.max_epoch = 120
    
    if hasattr(model, 'train_config'):
        model_train_config = model.train_config()
        for key, value in model_train_config.iteritems():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config

def train_BCE_AE(args, model, dataset):
    train_autoencoder(args, model, dataset, BCE_Loss=True)

def train_MSE_AE(args, model, dataset):
    train_autoencoder(args, model, dataset, BCE_Loss=False)

def train_autoencoder(args, model, dataset, BCE_Loss):
    if BCE_Loss:
        model.netid = "BCE." + model.netid
    else:
        model.netid = "MSE." + model.netid

    home_path = Models.get_ref_model_path(args, model.__class__.__name__, dataset.name, model_setup=True, suffix_str=model.netid)
    hbest_path = os.path.join(home_path, 'model.best.pth')
    hlast_path = os.path.join(home_path, 'model.last.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)

    if not os.path.isfile(hbest_path+".done"):
        config = get_ae_config(args, model, dataset, BCE_Loss=BCE_Loss)
        trainer = IterativeTrainer(config, args)
        print(colored('Training from scratch', 'green'))
        best_loss = 999999999
        for epoch in range(1, config.max_epoch+1):

            # Track the learning rates.
            lrs = [float(param_group['lr']) for param_group in config.optim.param_groups]
            config.logger.log('LRs', lrs, epoch)
            config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
            # One epoch of train and test.
            trainer.run_epoch(epoch, phase='train')
            trainer.run_epoch(epoch, phase='test')

            train_loss = config.logger.get_measure('train_loss').mean_epoch()
            test_loss = config.logger.get_measure('test_loss').mean_epoch()

            config.scheduler.step(train_loss)

            if config.visualize:
                # Show the average losses for all the phases in one figure.
                config.logger.visualize_average_keys('.*_loss', 'Average Loss', trainer.visdom)
                config.logger.visualize_average_keys('.*_accuracy', 'Average Accuracy', trainer.visdom)
                config.logger.visualize_average('LRs', trainer.visdom)

            # Save the logger for future reference.
            torch.save(config.logger.measures, os.path.join(home_path, 'logger.pth'))

            # Saving a checkpoint. Enable if needed!
            # if args.save and epoch % 10 == 0:
            #     print('Saving a %s at iter %s'%(colored('snapshot', 'yellow'), colored('%d'%epoch, 'yellow')))
            #     torch.save(config.model.state_dict(), os.path.join(home_path, 'model.%d.pth'%epoch))

            if args.save and test_loss < best_loss:
                print('Updating the on file model with %s'%(colored('%.4f'%test_loss, 'red')))
                best_loss = test_loss
                torch.save(config.model.state_dict(), hbest_path)
        
        torch.save({'finished':True}, hbest_path+".done")
        torch.save(config.model.state_dict(), hlast_path)

        if config.visualize:
            trainer.visdom.save([trainer.visdom.env])
    else:
        print("Skipping %s"%(colored(home_path, 'yellow')))

def train_variational_autoencoder(args, model, dataset):
    home_path = Models.get_ref_model_path(args, model.__class__.__name__, dataset.name, model_setup=True, suffix_str=model.netid)
    hbest_path = os.path.join(home_path, 'model.best.pth')
    hlast_path = os.path.join(home_path, 'model.last.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)

    if not os.path.isfile(hbest_path+".done"):
        config = get_vae_config(args, model, dataset)
        trainer = IterativeTrainer(config, args)
        print(colored('Training from scratch', 'green'))
        best_loss = 999999999
        for epoch in range(1, config.max_epoch+1):

            # Track the learning rates.
            lrs = [float(param_group['lr']) for param_group in config.optim.param_groups]
            config.logger.log('LRs', lrs, epoch)
            config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
            # One epoch of train and test.
            trainer.run_epoch(epoch, phase='train')
            trainer.run_epoch(epoch, phase='test')

            train_loss = config.logger.get_measure('train_loss').mean_epoch()
            test_loss = config.logger.get_measure('test_loss').mean_epoch()

            config.scheduler.step(train_loss)

            if config.visualize:
                # Show the average losses for all the phases in one figure.
                config.logger.visualize_average_keys('.*_loss', 'Average Loss', trainer.visdom)
                config.logger.visualize_average_keys('.*_accuracy', 'Average Accuracy', trainer.visdom)
                config.logger.visualize_average('LRs', trainer.visdom)

            # Save the logger for future reference.
            torch.save(config.logger.measures, os.path.join(home_path, 'logger.pth'))

            # Saving a checkpoint. Enable if needed!
            # if args.save and epoch % 10 == 0:
            #     print('Saving a %s at iter %s'%(colored('snapshot', 'yellow'), colored('%d'%epoch, 'yellow')))
            #     torch.save(config.model.state_dict(), os.path.join(home_path, 'model.%d.pth'%epoch))

            if args.save and test_loss < best_loss:
                print('Updating the on file model with %s'%(colored('%.4f'%test_loss, 'red')))
                best_loss = test_loss
                torch.save(config.model.state_dict(), hbest_path)
        
        torch.save({'finished':True}, hbest_path+".done")
        torch.save(config.model.state_dict(), hlast_path)

        if config.visualize:
            trainer.visdom.save([trainer.visdom.env])
    else:
        print("Skipping %s"%(colored(home_path, 'yellow')))
