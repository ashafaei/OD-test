import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import models as Models
import global_vars as Global
from utils.iterative_trainer import IterativeTrainer, IterativeTrainerConfig
from utils.logger import Logger
from datasets import MirroredDataset

from torch.utils.data import WeightedRandomSampler

from methods.logistic_threshold import KWayLogisticLoss, KWayLogisticWrapper

def get_KLclassifier_config(args, model, domain):
    print("Preparing training D1 for %s"%(domain.name))

    dataset = domain.get_D1_train()

    # 80%, 20% for local train+test
    train_ds, valid_ds = dataset.split_dataset(0.8)

    if domain.name in Global.mirror_augment:
        print("Mirror augmenting %s"%domain.name)
        new_train_ds = train_ds + MirroredDataset(train_ds)
        train_ds = new_train_ds

    #recalculate weighting
    class_weights = domain.calculate_D1_weighting()
    d1_set = train_ds
    weights = [0] * len(d1_set)                                              
    for idx, val in enumerate(d1_set):                                          
        weights[idx] = class_weights[val[1]]

    train_sampler = WeightedRandomSampler(weights, len(train_ds),replacement=False)

    # Initialize the multi-threaded loaders.
    pin = (args.device != 'cpu')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,  shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)
    all_loader   = DataLoader(dataset,  batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    # Set up the criterion
    criterion = KWayLogisticLoss()

    # Set up the model
    #klmodel = KWayLogisticWrapper(model).to(args.device)
    klmodel = KWayLogisticWrapper(model)

    # Set up the config
    config = IterativeTrainerConfig()

    config.name = 'KWayLogistic_%s_%s'%(domain.name, model.__class__.__name__)

    config.train_loader = train_loader
    config.valid_loader = valid_loader
    config.phases = {
                    'train':   {'dataset' : train_loader,  'backward': True},
                    'test':    {'dataset' : valid_loader,  'backward': False},
                    'all':     {'dataset' : all_loader,    'backward': False},                        
                    }
    config.criterion = criterion
    config.classification = True
    config.stochastic_gradient = True
    config.visualize = not args.no_visualize
    config.model = klmodel
    config.logger = Logger()

    config.optim = optim.Adam(klmodel.parameters(), lr=1e-3)
    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-2, min_lr=1e-6, factor=0.1, verbose=True)
    config.max_epoch = 120
    
    if hasattr(model, 'train_config'):
        model_train_config = model.train_config()
        for key, value in model_train_config.items():
            print('Overriding config.%s'%key)
            config.__setattr__(key, value)

    return config

def train_classifier(args, model, domain):
    config = get_KLclassifier_config(args, model, domain)

    home_path = Models.get_ref_model_path(args, model.__class__.__name__, domain.name, model_setup=True, suffix_str='KLogistic')
    hbest_path = os.path.join(home_path, 'model.best.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)

    trainer = IterativeTrainer(config, args)

    if not os.path.isfile(hbest_path+".done"):
        print('Training from scratch')
        best_accuracy = -1
        for epoch in range(1, config.max_epoch+1):

            # Track the learning rates.
            lrs = [float(param_group['lr']) for param_group in config.optim.param_groups]
            config.logger.log('LRs', lrs, epoch)
            config.logger.get_measure('LRs').legend = ['LR%d'%i for i in range(len(lrs))]
            
            # One epoch of train and test.
            trainer.run_epoch(epoch, phase='train')
            trainer.run_epoch(epoch, phase='test')

            train_loss = config.logger.get_measure('train_loss').mean_epoch()
            config.scheduler.step(train_loss)

            test_average_acc = config.logger.get_measure('test_accuracy').mean_epoch()

            # Save the logger for future reference.
            torch.save(config.logger.measures, os.path.join(home_path, 'logger.pth'))

            # Saving a checkpoint. Enable if needed!
            # if args.save and epoch % 10 == 0:
            #     print('Saving a %s at iter %s'%(colored('snapshot', 'yellow'), colored('%d'%epoch, 'yellow')))
            #     torch.save(config.model.state_dict(), os.path.join(home_path, 'model.%d.pth'%epoch))

            if args.save and best_accuracy < test_average_acc:
                print('Updating the on file model with %s'%('%.4f'%test_average_acc))
                best_accuracy = test_average_acc
                torch.save(config.model.state_dict(), hbest_path)
        
        torch.save({'finished':True}, hbest_path+".done")

    else:
        print("Skipping %s"%(home_path))

    print("Loading the best model.")
    config.model.load_state_dict(torch.load(hbest_path))
    config.model.eval()

    trainer.run_epoch(0, phase='all')
    test_average_acc = config.logger.get_measure('all_accuracy').mean_epoch(epoch=0)
    print("All average accuracy %s"%'%.4f%%'%(test_average_acc*100))
