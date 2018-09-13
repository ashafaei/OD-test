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

import models.pixelcnn.utils as pcnn_utils

def sample(model, batch_size, obs):
    model.train(False)
    nmix = model.nr_logistic_mix
    data = torch.zeros(batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    rescaling_inv = lambda x : .5 * x  + .5
    with torch.set_grad_enabled(False):
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = data
                out   = model(data_v, sample=True)
                out_sample = None
                if obs[0] == 1:
                    out_sample = pcnn_utils.sample_from_discretized_mix_logistic_1d(out, nmix)
                else:
                    out_sample = pcnn_utils.sample_from_discretized_mix_logistic(out, nmix)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return rescaling_inv(data)

def get_pcnn_config(args, model, dataset):
    print("Preparing training D1 for %s"%(dataset.name))

    sample_im, _ = dataset[0]
    obs = sample_im.size()
    obs = [int(d) for d in obs]
    
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
    criterion = pcnn_utils.PCNN_Loss(one_d=(model.input_channels==1))

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'PCNN_%s_%s'%(dataset.name, model.preferred_name())

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
    config.model = model
    config.logger = Logger()
    config.sampler = lambda x: sample(x.model, 32, obs)

    config.optim = optim.Adam(model.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-2, min_lr=1e-5, factor=0.1, verbose=True)
    config.max_epoch = 60
    
    if hasattr(model, 'train_config'):
        model_train_config = model.train_config()
        for key, value in model_train_config.iteritems():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config


def train_pixelcnn(args, model, dataset):
    home_path = Models.get_ref_model_path(args, model.__class__.__name__, dataset.name, model_setup=True, suffix_str=model.netid)
    hbest_path = os.path.join(home_path, 'model.best.pth')
    hlast_path = os.path.join(home_path, 'model.last.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)

    if not os.path.isfile(hbest_path+".done"):
        config = get_pcnn_config(args, model, dataset)
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
                config.logger.visualize_average('LRs', trainer.visdom)
                samples = config.sampler(config)
                trainer.visdom.images(samples.cpu(), win='sample_images')

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
