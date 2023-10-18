import torch.nn as nn
import torch
import numpy as np

def activation_helper(activation:str='relu'):
    activation = activation.lower()
    if activation == 'relu':
        activation = nn.ReLU
    elif activation == 'leakyrelu':
        activation = nn.LeakyReLU
    elif activation == 'sigmoid':
        activation = nn.Sigmoid
    else:
        raise NotImplementedError('Activation function {} not supported yet.'.format(activation))
    
    return activation

class logger:
    def __init__(self, name:str, dtype=float, mode:str='mean') -> None:
        if mode not in ['mean', 'sum', 'last', 'max', 'min']:
            raise NotImplementedError(f'Mode {mode} not supported yet.')
        
        self.name = name
        self.mode = mode
        self.dtype = dtype
        self.buffer = []
    
    def log(self, value):
        if not isinstance(value, self.dtype):
            raise TypeError(f'Value type {type(value)} not match logger type {self.dtype}.')
        self.buffer.append(value)
    
    def get_value(self, mode=None):
        if mode is None:
            mode = self.mode
            
        if mode == 'mean':
            out = sum(self.buffer) / len(self.buffer)
        elif mode == 'sum':
            out = sum(self.buffer)
        elif mode == 'max':
            out = max(self.buffer)
        elif mode == 'min':
            out = min(self.buffer)
        elif mode == 'last':
            out = self.buffer[-1]
        else:
            out = sum(self.buffer) / len(self.buffer)

        return out
    
    def clear(self):
        self.buffer = []
        
    def output(self, mode=None) -> str:
        out = self.get_value(mode)
        std = np.std(self.buffer)
        if mode == 'std':
            if self.dtype == int or self.dtype == str: 
                return f'{out}\u00B1{std}'
            else:
                return f'{out:.2f}\u00B1{std:.2f}'
        else:
            out = self.get_value(mode)
            if self.dtype == int or self.dtype == str: 
                return f'{out}'
            else:
                return f'{out:.2f}'
        
class logging_helper:
    def __init__(self):
        self.reset()

    def register(self, name:str, dtype=float, mode:str='last'):
        if name in self.log_dict.keys():
            raise ValueError(f'Name {name} already registered.')
        
        self.log_dict[name] = logger(name=name, dtype=dtype, mode=mode)
        
    def log(self, name:str, value):
        if name not in self.log_dict.keys():
            raise ValueError(f'Name {name} not registered.')
        
        self.log_dict[name].log(value)
    
    def get(self, name, mode=None):
        if name not in self.log_dict.keys():
            raise ValueError(f'Name {name} not registered.')
        
        return self.log_dict[name].output(mode)
    
    def step_output(self, name_list:list=None, epoch_name='Epoch', refresh=True):
        if epoch_name is not None:
            print(f'{epoch_name} {self.epoch_idx:3}>', end='\t')
        else:
            print('\t', end='\t')
            
        if name_list is None:
            # Default: output all registered names
            name_list = self.log_dict.keys()
        else:
            # Check if all names are registered
            for name in name_list:
                if name not in self.log_dict.keys():
                    raise ValueError(f'Name {name} not registered.')
        
        # Output
        for name in name_list:
            print(f'{name}: {self.log_dict[name].output("last")}', end='\t')
        
        # Whether return or newline
        if refresh:
            print('', end='\r')
        else:
            print('')
        # Step epoch
        self.epoch_idx += 1
    
    def final_output(self, name_list:list=None, std=False):
        print(f'Final>', end='\t')
        if name_list is None:
            # Default: output all registered names
            name_list = self.log_dict.keys()
        else:
            # Check if all names are registered
            for name in name_list:
                if name not in self.log_dict.keys():
                    raise ValueError(f'Name {name} not registered.')
        
        # Output
        for name in name_list:
            if std:
                print(f'{name}: {self.log_dict[name].output("std")}', end='\t')
            else:
                print(f'{name}: {self.log_dict[name].output()}', end='\t')
        print('')
        
    def reset(self):
        self.epoch_idx = 0
        self.log_dict = dict()
    
    def clear(self, name):
        self.log_dict[name].clear()

class earlystop_helper:
    '''
    Return 1 when value is better than best_value, -1 when early stop, 0 otherwise.
    '''
    
    def __init__(self, early_stop=10, mode='min'):
        self.count = 0
        self.early_stop = early_stop
        self.mode = mode
        if self.mode == 'min':
            self.best_value = torch.inf
        elif self.mode == 'max':
            self.best_value = -torch.inf
        else:
            raise NotImplementedError(f'Mode {mode} not supported yet.')
    
    def update(self, value):
        if self.mode == 'min' and value < self.best_value:
            self.best_value = value
            self.count = 0
            return 1
        elif self.mode == 'max' and value > self.best_value:
            self.best_value = value
            self.count = 0
            return 1
        else:
            self.count += 1
            if self.count >= self.early_stop:
                return -1
            else:
                return 0