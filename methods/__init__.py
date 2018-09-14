import torch
import torch.nn as nn

class AbstractMethodInterface(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def propose_H(self, dataset):
        raise NotImplementedError("%s does not have implementations for this"%(self.name))

    def train_H(self, dataset):
        raise NotImplementedError("%s does not have implementations for this"%(self.name))

    def test_H(self, dataset):
        raise NotImplementedError("%s does not have implementations for this"%(self.name))
    
    def method_identifier(self):
        raise NotImplementedError("Please implement the identifier method for %s"%(self.name))

class AbstractModelWrapper(nn.Module):
    def __init__(self, base_model):
        super(AbstractModelWrapper, self).__init__()
        self.base_model = base_model
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        if hasattr(self.base_model, 'parameters'):
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False
            
        self.eval_direct = False
        self.cache = {} #Be careful what you cache! You wouldn't have infinite memory.

    def set_eval_direct(self, eval_direct):
        self.eval_direct = eval_direct

    def train(self, mode=True):
        """ Must override the train mode 
        because the base_model is always in eval mode.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        # Now revert back the base_model to eval.
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        return self

    def subnetwork_eval(self, x):
        raise NotImplementedError

    def wrapper_eval(self, x):
        raise NotImplementedError

    def subnetwork_cached_eval(self, x, indices, group):
        output = None
        cache  = None

        if self.cache.has_key(group):
            cache = self.cache[group]
        else:
            cache = {}

        all_indices = [cache.has_key(ind) for ind in indices]
        if torch.ByteTensor(all_indices).all():
            # Then fetch from the cache.
            all_outputs = [cache[ind] for ind in indices]
            output = torch.cat(all_outputs)        
        else:
            output = self.subnetwork_eval(x)
            for i, entry in enumerate(output):
                cache[indices[i]] = entry.unsqueeze_(0)

        self.cache[group] = cache
        return output

    def forward(self, x, indices=None, group=None):
        input = None

        if not self.eval_direct:
            if indices is None:
                input = self.subnetwork_eval(x)
            else:
                input = self.subnetwork_cached_eval(x, indices=indices, group=group)
            input = input.detach()
            input.requires_grad = False
        else:
            input = x

        output = self.wrapper_eval(input)
        return output


class SVMLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SVMLoss, self).__init__()
        self.margin = margin
        self.size_average = True
    
    def forward(self, x, target):
        target = target.clone()
        # 0 labels should be set to -1 for this loss.
        target.data[target.data<0.1]=-1
        error = self.margin-x*target
        loss = torch.clamp(error, min=0)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def get_cached(model, dataset_loader, device):
    from tqdm import tqdm

    outputX, outputY = [], []
    with torch.set_grad_enabled(False):
        with tqdm(total=len(dataset_loader)) as pbar:
            pbar.set_description('Caching data')
            for i, (image, label) in enumerate(dataset_loader):
                pbar.update()
                input, target = image.to(device), label.to(device)
                new_input = model.subnetwork_eval(input)
                outputX.append(new_input)
                outputY.append(target)
    return torch.cat(outputX, 0), torch.cat(outputY, 0)
