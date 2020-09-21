# -*- coding: utf-8 -*-

"""
####################

# Network initialization

# Defines different initialization strategies for the networks

# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import math
import torch.nn as nn
import torch.nn.init as init

class InitializationStrategy(nn.Module):
    
    init_funcs = {
            'uniform':init.uniform_,
            'normal':init.normal_,
            'eye':init.eye_,
            'dirac':init.dirac_,
            'xavier':init.xavier_normal_,
            'xavier_uniform':init.xavier_uniform_,
            'xavier_normal':init.xavier_normal_,
            'kaiming':init.kaiming_normal_,
            'kaiming_uniform':init.kaiming_uniform_,
            'kaiming_normal':init.kaiming_normal_,
            'orthogonal':init.orthogonal_,
            'sparse':init.sparse_
            }
    
    def __init__(self, args):
        super(InitializationStrategy, self).__init__()
        if (args.initialize == 'classic'):
            self.init_pointer = self.init_classic
        else:
            self.init_func = self.init_funcs[args.initialize]
            self.init_pointer = self.init_function
        
    # Function for Initialization
    def init_classic(self, m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        #if m.__class__ in [nn.Conv1d, nn.ConvTranspose1d]:
        #    init.xavier_normal_(m.weight.data)
        #    if m.bias is not None:
        #        init.normal_(m.bias.data)
        if m.__class__ in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif m.__class__ in [nn.Linear]:
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif m.__class__ in [nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell]:
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
                    
    # Function for Initialization
    def init_function(self, m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if m.__class__ in [nn.Conv1d, nn.ConvTranspose1d]:
            m.reset_parameters()
            #init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
            #if m.bias is not None:
            #    init.normal_(m.bias.data)
        if m.__class__ in [nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
            self.init_func(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        if m.__class__ in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif m.__class__ in [nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell]:
            for param in m.parameters():
                if len(param.shape) >= 2:
                    self.init_func(param.data)
                #else:
                #    self.init_func(param.data)
    
    def __call__(self, m):
        self.init_pointer(m)