"""
    This is an implementation of OpenMax from
    A. Bendale, T. Boult [Towards Open Set Deep Networks](http://vast.uccs.edu/~abendale/papers/0348.pdf). CVPR 2016
    The implementation is checked against the original implementation https://github.com/abhijitbendale/OSDN
    we do not rely on third party libraries to perform the calculations. Everything that is necessary for this
    is already implemented here from scratch. 

    Notes:
        - The original method was tested with AlexNet, as with all other methods we test this with ResNet and VGG.
        - Using the terminology of the paper, here, we have only one channel. But technically, the authors
            meant batch_size rather than "channels". The batches came from random crops in the Caffe python sample code.
    
    This is not an efficient implementation. Specifically, OTModelWrapper.subnetwork_eval can be improved.
    We decided to keep the structure to the original source code as much as possible.
"""
from __future__ import print_function

import numpy as np
from scipy.stats import exponweib
from tqdm import tqdm
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from methods import AbstractMethodInterface, AbstractModelWrapper, SVMLoss
from methods.base_threshold import ProbabilityThreshold
from utils.iterative_trainer import IterativeTrainerConfig
from utils.logger import Logger
from datasets import MirroredDataset
import global_vars as Global

import scipy.stats

class LibNotMR(object):
    """
        Instead of using LibMR (https://github.com/abhijitbendale/OSDN/tree/master/libMR) we implemented
        the simple operations with Scipy. The output is checked against the original library for verification.
    """
    def __init__(self, tailsize = 20):
        self.tailsize = tailsize
        self.min_val = None
        self.translation = 10000 # this constant comes from the library.
                                 # it only makes small numerical differences.
                                 # we keep it for authenticity.
        self.a = 1
        self.loc = 0
        self.c = None
        self.scale = None

    def fit_high(self, inputs):
        inputs = inputs.numpy()
        tailtofit = sorted(inputs)[-self.tailsize:]
        self.min_val = np.min(tailtofit)
        new_inputs = [i + self.translation - self.min_val for i in tailtofit]
        params = scipy.stats.exponweib.fit(new_inputs, floc=0, f0=1)
        self.c = params[1]
        self.scale = params[3]
    
    def w_score(self, inputs):
        new_inputs = inputs + self.translation - self.min_val
        new_score = scipy.stats.exponweib.cdf(new_inputs, a=self.a, c=self.c, loc=self.loc, scale=self.scale)
        return new_score
    
    def serialize(self):
        return torch.FloatTensor([self.min_val, self.c, self.scale])
    
    def deserialize(self, params):
        self.min_val = params[0].item()
        self.c = params[1].item()
        self.scale = params[2].item()
    
    def __str__(self):
        return 'Weib: C=%.2f scale=%.2f min_val=%.2f'%(self.c, self.scale, self.min_val)

def distance_measure(ref, target):
    ref     = ref.unsqueeze(0)
    target  = target.unsqueeze(0)

    euc_dist = torch.pairwise_distance(ref, target)
    cos_dist = 1 - torch.nn.functional.cosine_similarity(ref, target)
    query_dist = euc_dist.item()/200.0 + cos_dist.item() # the coefficients are taken from the original implementation.
    return query_dist

class OTModelWrapper(AbstractModelWrapper):
    """ The wrapper class for H.
    """
    def __init__(self, base_model, mav, mr_models, alpharank = 10):
        super(OTModelWrapper, self).__init__(base_model)

        # Serialize the MR models.
        n_classes = mav.size(0)
        weib_params = torch.FloatTensor(n_classes, 3).fill_(0)
        for cl in range(n_classes):
            weib_params[cl] = mr_models[cl].serialize()

        self.H = nn.Sequential(
                    nn.BatchNorm1d(n_classes+1), # Helps with faster convergence.
                    nn.Linear(n_classes+1, 1),
        )
        self.H.register_buffer('MAV', mav.clone())
        self.H.register_buffer('mr_params', weib_params.clone())
        self.alpharank = alpharank
        self.reload_mr()
    
    def reload_mr(self):
        self.mr_models = []
        n_classes = self.H.MAV.size(0)
        for cl in range(n_classes):
            mr_model = LibNotMR()
            mr_model.deserialize(self.H.mr_params[cl])
            self.mr_models.append(mr_model)

    def set_eval_direct(self, eval_direct):
        super(OTModelWrapper, self).set_eval_direct(eval_direct)
        self.reload_mr()
    
    def eval(self):
        super(OTModelWrapper, self).eval()
        self.reload_mr()

    def subnetwork_eval(self, x):
        base_output = self.base_model(x, softmax=False)

        n_instances, n_classes = base_output.size()
        alpharank = self.alpharank

        output = base_output.new(n_instances, n_classes + 1).fill_(0)
        for i in range(n_instances):
            openmax_activation = []
            openmax_unknown = []
            activation = base_output[i]

            alpha_weights = [((alpharank+1) - t)/float(alpharank) for t in range(1, alpharank+1)]
            ranked_alpha = base_output.new_zeros(n_classes)
            _, ranked_list = activation.sort()
            for k in range(len(alpha_weights)):
                ranked_alpha[ranked_list[k]] = alpha_weights[k]
            for j in range(n_classes):
                query_dist = distance_measure(self.H.MAV[j, :], activation)
                wscore = base_output.new([self.mr_models[j].w_score(query_dist)])
                modified_score = activation[j] * ( 1 - wscore*ranked_alpha[j] )
                openmax_activation += [modified_score]
                openmax_unknown += [activation[j] - modified_score]
            openmax_activation = base_output.new(openmax_activation)
            openmax_unknown = base_output.new(openmax_unknown)

            activation_exp = openmax_activation.exp()
            total_sum = activation_exp.sum().item() + openmax_unknown.sum().exp().item()

            probs = activation_exp / total_sum
            unks  = openmax_unknown.sum().exp() / total_sum
            output[i, 0:n_classes] = probs
            output[i, n_classes] = unks

        return output.detach()

    def wrapper_eval(self, x):
        output = self.H(x)
        return output
    
    def classify(self, x):
        return (x > 0).long()


class OpenMax(ProbabilityThreshold):
    def __init__(self, args):
        super(OpenMax, self).__init__(args)
        self.tailsize = 20 # default in the paper.

    def method_identifier(self):
        output = "OpenMax"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def propose_H(self, dataset):
        # Do the PBThresh propose_h first.
        super(OpenMax, self).propose_H(dataset)

        """
            For each class we calculate the mean of logits over _correct classifications_.
        """
        n_classes = self.base_model.output_size()[1].item()
        mav = torch.FloatTensor(n_classes, n_classes).fill_(0).to(self.args.device)
        mav_count = torch.LongTensor(n_classes).fill_(0).to(self.args.device)
        data_loader = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=self.args.workers, pin_memory=True)

        with torch.set_grad_enabled(False):
            with tqdm(total=len(data_loader)) as pbar:
                pbar.set_description('Calculating MAV')
                for i, (image, label) in enumerate(data_loader):
                    pbar.update()
                    input, target = image.to(self.args.device), label.to(self.args.device)
                    prediction = self.base_model.forward(input, softmax=False)
                    _, max_ind = torch.max(prediction, dim=1)
                    for sub_ind in range(len(image)):
                        if max_ind[sub_ind].item() == label[sub_ind].item():
                            mav[label[sub_ind], :].add_(prediction[sub_ind])
                            mav_count[label[sub_ind]].add_(1)
        assert (mav_count>0).all().item() == 1, 'Something wrong with the classes! Need at least one sample/class.'
        mav.div_(mav_count.float().unsqueeze(1).expand_as(mav))
        self.mav = mav # Store the MAV.

        # In the original source code calculates three different distance measures
        # Cosine, Euclidean, and a combination, however, the default mode of 
        # distance is set to the combination and the other two are not used.
        distance_values = [[] for i in range(n_classes)]
        with torch.set_grad_enabled(False):
            with tqdm(total=len(data_loader)) as pbar:
                pbar.set_description('Calculating the distances')
                for i, (image, label) in enumerate(data_loader):
                    pbar.update()
                    input, target = image.to(self.args.device), label.to(self.args.device)
                    prediction = self.base_model.forward(input, softmax=False)
                    _, max_ind = torch.max(prediction, dim=1)
                    for sub_ind in range(len(image)):
                        if max_ind[sub_ind].item() == label[sub_ind].item():
                            query_dist = distance_measure(mav[label[sub_ind], :], prediction[sub_ind])
                            distance_values[label[sub_ind]].append(query_dist)
        torch_values = [torch.FloatTensor(dv) for dv in distance_values]
        
        self.weib_models = []
        with tqdm(total=n_classes) as pbar:
            pbar.set_description('Learning the Weibull model')
            for cl in range(n_classes):
                pbar.update()
                mr_model = LibNotMR(tailsize = self.tailsize)
                mr_model.fit_high(torch_values[cl])
                self.weib_models.append(mr_model)

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
        # margin must be non-zero.
        criterion = SVMLoss(margin=1.0).cuda()

        # Set up the model
        model = OTModelWrapper(self.base_model, self.mav, self.weib_models).to(self.args.device)

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
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-1, weight_decay=1.0/len(train_ds))
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=10, threshold=1e-1, min_lr=1e-8, factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100

        return config



