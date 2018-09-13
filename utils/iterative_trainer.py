from __future__ import print_function
import errno

import timeit
from tqdm import tqdm
from termcolor import colored

import torch
import torch.nn.functional as F
from visdom import Visdom

class IterativeTrainerConfig(object):
    pass

class IterativeTrainer(object):
    def __init__(self, config, args):
        self.config = config
        self.args   = args
        self.device = args.device
        if config.visualize:
            self.visdom = Visdom(ipv6=False, env='(%s)-%s:%s'%(args.hostname, args.experiment_id, config.name))
        # Set the default behaviours if not set.
        defaults = {
            'classification': False,
            'cast_float_label': False,
            'autoencoder_target': False,
            'autoencoder_class': False,
            'stochastic_gradient': True,
            'visualize': True,
            'sigmoid_viz': True,
        }
        for key, value in defaults.iteritems():
            if not hasattr(self.config, key):
                print(colored('Setting default value %s to %s'%(key, value), 'red'))
                setattr(self.config, key, value)
    
    def run_epoch(self, epoch, phase='train'):
        # Retrieve the appropriate config.
        config      = self.config.phases[phase]
        dataset     = config['dataset']
        backward    = config['backward']
        phase_name  = phase
        print("Doing %s"%colored(phase, 'green'))

        model       = self.config.model
        visualize   = self.config.visualize
        criterion   = self.config.criterion
        optimizer   = self.config.optim
        logger      = self.config.logger
        stochastic  = self.config.stochastic_gradient
        classification = self.config.classification

        # See the network to the target mode.
        if backward:
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        start_time = timeit.default_timer()
        last_viz_update = start_time

        # For full gradient optimization we need to rescale the loss
        # to calculate the gradient correctly.
        loss_scaler = 1
        if not stochastic:
            loss_scaler = 1./len(dataset.dataset)

        try:
            # TQDM sometimes throws IOError exceptions when you
            # try to close it. We ignore those exceptions.
            with tqdm(total=len(dataset)) as pbar:
                if backward and not stochastic:
                    optimizer.zero_grad()

                for i, (image, label) in enumerate(dataset):
                    pbar.update()
                    if backward and stochastic:
                        optimizer.zero_grad()

                    # Get and prepare data.
                    input, target, data_indices = image, None, None
                    if torch.typename(label) == 'list':
                        assert len(label) == 2, 'There should be two entries in the label'
                        # Need to unpack the label. This is for when the data provider
                        # has the cached flag enabled, therefore the y is now (y, idx).
                        target, data_indices = label
                    else:
                        target = label
                    
                    if self.config.autoencoder_target:
                        target = input.clone()
                    
                    if self.config.cast_float_label:
                        target = target.float().unsqueeze(1)

                    input, target = input.to(self.device), target.to(self.device)

                    # Do a forward propagation and get the loss.
                    prediction = None
                    if data_indices is None:
                        prediction = model(input)
                    else:
                        # Run in the cached mode. This is necessary to speed up
                        # some of the underlying optimization procedures. It is not
                        # always used though.
                        prediction = model(input, indices=data_indices, group=phase_name)

                    loss = criterion(prediction, target)

                    if backward:
                        if stochastic:
                            loss.backward()
                            optimizer.step()
                        else:
                            nscaler = loss_scaler
                            if criterion.size_average:
                                nscaler = nscaler * len(input)
                            loss2 = loss * nscaler
                            loss2.backward()

                    # Compute various measure. Can be safely skipped.
                    if not backward or not stochastic:
                        if criterion.size_average:
                            loss.data.mul_(len(input))
                    logger.log('%s_loss'%phase_name, loss.item(), epoch, i)
                    message = '%s Batch loss %.3f'%(phase_name, loss.item())

                    if classification:
                        pred = []
                        if prediction.size(1) == 1:
                            # For binary classification, we do this to make the
                            # prediction code consistent with the max.
                            pred = model.classify(prediction)
                        else:
                            pred = prediction.max(1)[1]
                        if stochastic and backward:
                            acc = (pred == target.long()).float().view(-1).mean().item()
                            logger.log('%s_accuracy'%phase_name, acc, epoch, i)
                        else:
                            acc = (pred == target.long()).float().view(-1).sum().item()
                            logger.log('%s_accuracy'%phase_name, acc, epoch, i)
                            acc = acc/target.numel()
                        message = '%s Accuracy %.2f'%(message, acc)

                    pbar.set_description(message)

                    if visualize and i % 20 == 0 and (timeit.default_timer() - last_viz_update > 5):
                        # we don't want to update too quickly. visdom breaks!
                        logger.visualize_epoch('%s_loss'%phase_name, self.visdom)
                        self.visdom.images(image.numpy(), win='in_images')
                        if self.config.autoencoder_target and prediction.size(1) in [1, 3]:
                            viz_ten = None
                            if self.config.sigmoid_viz:
                                viz_ten = F.sigmoid(prediction).cpu().detach().numpy()
                            else:
                                viz_ten = prediction.cpu().detach().numpy()
                            self.visdom.images(viz_ten, win='out_images')
                        last_viz_update = timeit.default_timer()

                if backward and not stochastic:
                    optimizer.step()
        except IOError, e:
            if e.errno != errno.EINTR:
                raise
            else:
                print(colored("Problem averted :D", 'green'))

        if not backward or not stochastic:
            logger.get_measure('%s_loss'%phase_name).measure_normalizer = len(dataset.dataset)
            if classification:
                logger.get_measure('%s_accuracy'%phase_name).measure_normalizer = len(dataset.dataset)

        elapsed = timeit.default_timer() - start_time
        print('  %s Epoch %d in %.2fs' %(phase_name, epoch, elapsed))
