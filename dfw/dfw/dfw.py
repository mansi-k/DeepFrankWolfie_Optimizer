import torch
import torch.optim as optim

from collections import defaultdict

'''Making DFW a child class of optim.Optimizer'''
class DFW(optim.Optimizer):
    '''
    Args:
        params: iterable that contains parameters for optimization
        eta : initial learning rate (float)
        momentum : optional momentum factor (float) (default 0)
        weight_decay : optional L2 penalty for regularization (default 0)
        eps : tiny constant to avoid divide by zero error (default 1e-5)
    '''

    def __init__(self, params, eta, weight_decay=0, momentum=0, eps=1e-5):
        if eta <= 0.0:
            raise ValueError("Invalid eta")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay")
        if momentum < 0.0:
            raise ValueError("Invalid momentum")
        
        '''Creating a dictionary with default values required by the parent class Optimizer'''
        defaults = dict(eta=eta, momentum=momentum, weight_decay=weight_decay)
        '''Initializing parent class variables'''
        super(DFW, self).__init__(params, defaults)
        
        self.eps = eps
        
        '''param_groups and state are members of parent class Optimizer'''
        '''Momentum will be accumulated for each parameter, thus initializing it with zero tensor of size same as the parameter tensor'''
        for grp in self.param_groups:
            if grp['momentum']:
                for prm in grp['params']:
                    self.state[prm]['accumulated_momentum'] = torch.zeros_like(prm.data, requires_grad=False)

    @torch.autograd.no_grad()
    def step(self, closure):
        '''This function performs a single optimization step'''
        '''Closure here is a lambda function that returns float value of the loss as DFW needs to have access to the current loss value'''
        cur_loss = float(closure())
        
        '''Creating a dictionary for collecting gradients and updating weights'''
        weights_dict = defaultdict(dict)
        for grp in self.param_groups:
            w_decay = grp['weight_decay']
            for prm in grp['params']:
                if prm.grad is not None:
                    '''Collecting derivatives/gradients'''
                    weights_dict[prm]['delta_t'] = prm.grad.data
                    '''Applying L2 penalty (weight decay) to the weights'''
                    weights_dict[prm]['r_t'] = w_decay * prm.data

        '''Updating gamma (member of parent class) with new optimal step size'''
        self.gamma = self._cal_step_size(cur_loss, weights_dict)

        for grp in self.param_groups:
            eta = grp['eta']
            mu = grp['momentum']
            for prm in grp['params']:
                if prm.grad is not None:
                    state = self.state[prm]
                    delta_t, r_t = weights_dict[prm]['delta_t'], weights_dict[prm]['r_t']
                    '''Performing parameter update on original weights'''
                    prm.data -= eta * (r_t + self.gamma * delta_t)
                    '''Applying momentum (if exists) to the parameter weights according to eq(45) in the paper'''
                    if mu:
                        z_t = state['accumulated_momentum']
                        z_t *= mu
                        z_t -= eta * self.gamma * (delta_t + r_t)
                        prm.data += mu * z_t

    @torch.autograd.no_grad()
    def _cal_step_size(self, cur_loss, weights_dict):
        '''This function computes the line search in closed form for calculating optimal step size'''
        denominator = self.eps
        loss = cur_loss

        for grp in self.param_groups:
            eta = grp['eta']
            for prm in grp['params']:
                if prm.grad is not None:
                    delta_t = weights_dict[prm]['delta_t']
                    r_t = weights_dict[prm]['r_t']
                    '''Optimal step_size calculation formula as given in paper : eq10 in Algorithm1'''
                    loss -= eta * torch.sum(delta_t * r_t)
                    denominator += eta * delta_t.norm()**2

        '''Returning optimal step size'''
        return float((loss / denominator).clamp(min=0, max=1))
