from __future__ import print_function
from os import path
from termcolor import colored

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import global_vars as Global
import models as Models
from datasets import MirroredDataset
from methods.score_svm import ScoreSVM

from tqdm import tqdm

class KNNModel(nn.Module):
    """
        This is our Nearest Neighbour "neural network".
    """

    def __init__(self, base_data, k=1):
        super(KNNModel, self).__init__()
        self.base_data = base_data.half().cuda()         # We probably have to rewrite this part of the code 
                                                         # as larger datasets may not entirely fit into the GPU memory.
        n_data = self.base_data.size(0)
        self.base_data = self.base_data.view(n_data, -1) # Flatten the train data.
        self.base_data_norm = (self.base_data*self.base_data).sum(dim=1)
        self.K = k
        self.norm = 2
    
    def forward(self, x, **kwargs):
        n_samples = x.size(0)
        x = x.data.view(n_samples, -1).half() # flatten to vectors.
        base_data = self.base_data
        base_norm = self.base_data_norm
        ref_size = base_data.size(0)

        x_norm = (x*x).sum(dim=1)
        diffs = base_norm.unsqueeze(0).expand(n_samples, ref_size) + x_norm.unsqueeze(1).expand(n_samples, ref_size) - 2*x.matmul(base_data.t())
        diffs.sqrt_().detach_()

        output, _ = torch.topk(diffs, self.K, dim=1, largest=False, sorted=True)

        return output.float()
    
    def preferred_name(self):
        return '%d-NN'%self.K

    def output_size(self):
        return torch.LongTensor([1, self.K])

class KNNSVM(ScoreSVM):
    def __init__(self, args):
        super(KNNSVM, self).__init__(args)
        self.base_data = None
        self.default_model = 1

    def method_identifier(self):
        output = "KNNSVM/%d"%self.default_model
        return output

    def propose_H(self, dataset):
        assert self.default_model > 0, 'KNN needs K>0'
        if self.base_model is not None:
            self.base_model.base_data = None
            self.base_model = None

        if dataset.name in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%dataset.name, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds
        
        n_data = len(dataset)
        n_dim  = dataset[0][0].numel()
        self.base_data = torch.zeros(n_data, n_dim, dtype=torch.float32)

        with tqdm(total=n_data) as pbar:
            pbar.set_description('Caching X_train for %d-nn'%self.default_model)
            for i, (x, _) in enumerate(dataset):
                self.base_data[i].copy_(x.view(-1))
                pbar.update()
        # self.base_data = torch.cat([x.view(1, -1) for x,_ in dataset])
        self.base_model = KNNModel(self.base_data, k=self.default_model).to(self.args.device)
        self.base_model.eval()

class AEKNNModel(nn.Module):
    """
        This is our Nearest Neighbour "neural network" with AE latent representations.
    """

    def __init__(self, subnetwork, base_data, k=1):
        super(AEKNNModel, self).__init__()
        self.base_data = base_data.cuda()
        n_data = self.base_data.size(0)
        self.base_data = self.base_data.view(n_data, -1) # Flatten the train data.
        self.base_data_norm = (self.base_data*self.base_data).sum(dim=1)
        self.K = k
        self.norm = 2
        self.subnetwork = subnetwork
    
    def forward(self, x, **kwargs):
        n_samples = x.size(0)
        self.subnetwork.eval()
        x = self.subnetwork.encode(x).data
        base_data = self.base_data
        base_norm = self.base_data_norm
        ref_size = base_data.size(0)

        x_norm = (x*x).sum(dim=1)
        diffs = base_norm.unsqueeze(0).expand(n_samples, ref_size) + x_norm.unsqueeze(1).expand(n_samples, ref_size) - 2*x.matmul(base_data.t())
        diffs.sqrt_().detach_()

        output, _ = torch.topk(diffs, self.K, dim=1, largest=False, sorted=True)

        return output.float()
    
    def preferred_name(self):
        return '%d-AENN'%self.K

    def output_size(self):
        return torch.LongTensor([1, self.K])

class AEKNNSVM(ScoreSVM):
    def __init__(self, args):
        super(AEKNNSVM, self).__init__(args)
        self.base_data = None
        self.default_model = 1

    def method_identifier(self):
        output = "AEKNNSVM/%d"%self.default_model
        return output

    def propose_H(self, dataset):
        assert self.default_model > 0, 'KNN needs K>0'
        if self.base_model is not None:
            self.base_model.base_data = None
            self.base_model = None

        # Set up the base-model
        if isinstance(self, BCEKNNSVM) or isinstance(self, MSEKNNSVM):
            base_model = Global.get_ref_autoencoder(dataset.name)[0]().to(self.args.device)
            if isinstance(self, BCEKNNSVM):
                base_model.netid = "BCE." + base_model.netid
            else:
                base_model.netid = "MSE." + base_model.netid
            home_path = Models.get_ref_model_path(self.args, base_model.__class__.__name__, dataset.name, suffix_str=base_model.netid)
        elif isinstance(self, VAEKNNSVM):
            base_model = Global.get_ref_vae(dataset.name)[0]().to(self.args.device)
            home_path = Models.get_ref_model_path(self.args, base_model.__class__.__name__, dataset.name, suffix_str=base_model.netid)
        else:
            raise NotImplementedError()

        hbest_path = path.join(home_path, 'model.best.pth')
        best_h_path = hbest_path
        print(colored('Loading H1 model from %s'%best_h_path, 'red'))
        base_model.load_state_dict(torch.load(best_h_path))
        base_model.eval()

        if dataset.name in Global.mirror_augment:
            print(colored("Mirror augmenting %s"%dataset.name, 'green'))
            new_train_ds = dataset + MirroredDataset(dataset)
            dataset = new_train_ds

        # Initialize the multi-threaded loaders.
        all_loader   = DataLoader(dataset,  batch_size=self.args.batch_size, num_workers=1, pin_memory=True)

        n_data = len(dataset)
        n_dim  = base_model.encode(dataset[0][0].to(self.args.device).unsqueeze(0)).numel()
        print('nHidden %d'%(n_dim))
        self.base_data = torch.zeros(n_data, n_dim, dtype=torch.float32)
        base_ind = 0
        with torch.set_grad_enabled(False):
            with tqdm(total=len(all_loader)) as pbar:
                pbar.set_description('Caching X_train for %d-nn'%self.default_model)
                for i, (x, _) in enumerate(all_loader):
                    n_data = x.size(0)
                    output = base_model.encode(x.to(self.args.device)).data
                    self.base_data[base_ind:base_ind+n_data].copy_(output)
                    base_ind = base_ind + n_data
                    pbar.update()
        # self.base_data = torch.cat([x.view(1, -1) for x,_ in dataset])
        self.base_model = AEKNNModel(base_model, self.base_data, k=self.default_model).to(self.args.device)
        self.base_model.eval()

"""
    Actual implementation in AEKNNSVM.
"""
class BCEKNNSVM(AEKNNSVM):
    def method_identifier(self):
        output = "BCEKNNSVM/%d"%self.default_model
        return output
class MSEKNNSVM(AEKNNSVM):
    def method_identifier(self):
        output = "MSEKNNSVM/%d"%self.default_model
        return output
class VAEKNNSVM(AEKNNSVM):
    def method_identifier(self):
        output = "VAEKNNSVM/%d"%self.default_model
        return output