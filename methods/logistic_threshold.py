from __future__ import print_function
from os import path
from termcolor import colored

import torch
import torch.nn as nn

import models as Models
from methods.score_svm import ScoreSVM
from utils.iterative_trainer import IterativeTrainer

class KWayLogisticWrapper(nn.Module):
    """
        This class wraps around classifiers and forces them to become
        a K-Way logistic regression model.
    """
    def __init__(self, parent_model):
        super(KWayLogisticWrapper, self).__init__()
        self.model = parent_model
    
    def forward(self, x, **kwargs):
        if kwargs.has_key('softmax'):
            del kwargs['softmax']
        model_output = self.model(x, softmax=False, **kwargs)
        return model_output
    
    def preferred_name(self):
        return self.model.__class__.__name__

    def output_size(self):
        return self.model.output_size()

class KWayLogisticLoss(nn.Module):
    def __init__(self):
        super(KWayLogisticLoss, self).__init__()
        self.size_average = True
        self.loss = nn.BCEWithLogitsLoss(size_average=True)
        
    def forward(self, input, target):
        n_classes = input.size(1)
        n_samples = input.size(0)
        target_expansion = input.new(n_samples, n_classes).zero_()
        target_expansion.scatter_(1, target.data.unsqueeze(1), 1)
        target_expansion.requires_grad = False
        loss = self.loss(input, target_expansion)
        return loss

class LogisticSVM(ScoreSVM):
    def method_identifier(self):
        output = "LogisticSVM"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        # Wrap the class in KWLWrapper
        original_class_name = config.model.__class__.__name__
        config.model = KWayLogisticWrapper(config.model)
        config.model = config.model.to(self.args.device)

        h_path = Models.get_ref_model_path(self.args, original_class_name, dataset.name, suffix_str='KLogistic')
        best_h_path = path.join(h_path, 'model.best.pth')
        
        trainer = IterativeTrainer(config, self.args)

        if not path.isfile(best_h_path):      
            raise NotImplementedError("Please use setup_model to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s'%best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))
        
        trainer.run_epoch(0, phase='all')
        test_average_acc = config.logger.get_measure('all_accuracy').mean_epoch(epoch=0)
        print("All average accuracy %s"%colored('%.4f%%'%(test_average_acc*100), 'red'))

        self.base_model = config.model
        self.base_model.eval()
