# -*- coding: utf-8 -*-

"""
####################

# Analysis functions

# Defines the basic operations for analyzing the resulting subnetworks

# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import os
import copy
import time
import torch
import torch.nn as nn
from statistics import ModelStat


class Analyzer(nn.Module):
    
    def __init__(self, args):
        super(Analyzer, self).__init__()
        self.current_it = 0
        self.prune_it = args.prune_it
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.model_save = args.model_save
        self.device = args.device
        self.output = args.output
        self.model_checkpoints = [None] * self.prune_it
        self.pruning_structure = [None] * self.prune_it
        self.model_stats = [None] * self.prune_it
        self.layers_data = [None] * self.prune_it
        self.iterations = [None] * self.prune_it
        self.losses = [None] * self.prune_it
        self.training_times = [None] * self.prune_it
        self.stopped_epoch = [None] * self.prune_it
        self.args = args

    def iteration_start(self, model, prune_it):
        self.current_it = prune_it
        # Save model checkpoint
        if (self.args.light_stats == 0):
            self.save_model_checkpoint(model, prune_it, 'initial')
        # Create a summary of the model
        self.model_summary(model, prune_it)
        # Record starting time
        self.training_times[prune_it] = time.time()

    def iteration_end(self, model, prune_it, stop_epoch, losses):
        # Save model checkpoint
        if (self.args.light_stats == 0):
            self.save_model_checkpoint(model, prune_it, 'final')
        # Append the losses curves of the current model
        self.losses[prune_it] = losses
        # Compute actual training time
        self.training_times[prune_it] = ((time.time() - self.training_times[prune_it]) / 60.0)
        # Early stopped epoch
        self.stopped_epoch[prune_it] = stop_epoch
        
    # Layer Looper
    def print_model(self, model):
        for name, param in model.named_parameters():
            print(name, param.size())
        
    def check_model_size(self, model):
        model.n_parameters()
        
    def model_summary(self, model, it):
        model_stats = {}
        stats = ModelStat(model, self.input_size)
        collected_nodes = stats._analyze_model()
        layers_data = list()
        layers = list()
        for node in collected_nodes:
            name = node.name
            input_shape = node.input_shape
            output_shape = node.output_shape
            parameter_quantity = node.parameter_quantity
            inference_memory = node.inference_memory
            layer_flops = node.Flops
            mread, mwrite = [i for i in node.Memory]
            duration = node.duration
            layers_data.append([parameter_quantity, inference_memory, duration, layer_flops, mread, mwrite])
            layers.append([name, input_shape, output_shape])
        layers_data = torch.tensor(layers_data)
        # Number of parameters
        model_stats['parameters'] = torch.sum(layers_data[:, 0]).item()
        # Memory used in saving
        model_stats['disk_size'] = torch.sum(layers_data[:, 1]).item()
        # Number of Flops
        model_stats['flops'] = torch.sum(layers_data[:, 3]).item()
        # Number of Flops
        model_stats['memory_read'] = torch.sum(layers_data[:, 4]).item()
        # Number of Flops
        model_stats['memory_write'] = torch.sum(layers_data[:, 5]).item()
        self.model_stats[it] = model_stats
        if (self.args.light_stats == 0):
            self.layers_data[it] = layers_data
        
    def register_pruning(self, model, pruning, it):
        # Export model to a file and keep path
        pruning_name = self.model_save + '/it_' + str(it) + '_pruning.th'
        if (self.args.prune == 'masking'):
            pruning_vals = pruning.mask
        elif (self.args.prune == 'trimming'):
            pruning_vals = []
            for (name, m) in pruning.leaf_modules:
                if (hasattr(m, 'unprune_idx')):
                    pruning_vals.append((name, m.unprune_idx))
        elif (self.args.prune == 'hybrid'):
            pruning_vals = []
            for (name, m) in pruning.leaf_modules:
                if (hasattr(m, 'unprune_idx')):
                    pruning_vals.append((name, m.unprune_idx))
            pruning_vals.append(pruning.mask)
        else:
            print('Undefined analysis behavior for ' + self.args.prune)
            exit(0)
        torch.save(pruning_vals, pruning_name)
        self.pruning_structure[it] = pruning_name
        
    def print_summary(self, model_stats=None):
        if (model_stats is None):
            model_stats = self.model_stats[self.current_it]
        summary = "=" * 32
        summary += '\n'
        summary += "Total params      : {:,}\n".format(int(model_stats['parameters']))
        summary += "-" * 32
        summary += '\n'
        summary += "Total memory      : {:.2f} MB\n".format(model_stats['disk_size'])
        summary += "Total Flops       : {}Flops\n".format(round_value(model_stats['flops']))
        summary += "Total Mem (Read)  : {}B\n".format(round_value(model_stats['memory_read'], True))
        summary += "Total Mem (Write) : {}B".format(round_value(model_stats['memory_write'], True))
        print(summary)
        
    def save_model_checkpoint(self, model, it, type_m='initial'):
        # Save the current model weights
        state_dict = copy.deepcopy(model) # model.state_dict())
        # Export model to a file and keep path
        model_check_name = self.model_save + '/it_' + str(it) + '_' + type_m + '.th'
        torch.save(state_dict, model_check_name)
        # Add to a list of checkpoints
        if (self.model_checkpoints[it] is None):
            self.model_checkpoints[it] = {}
        self.model_checkpoints[it][type_m] = model_check_name
        # Check resulting size
        file_info = os.stat(model_check_name)
        print('[Saved checkpoint - Size = ' + convert_bytes(file_info.st_size) + ']')
        
    def test_interpolations(self):
        print('Nope nothing done here ^^')
        
        
def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.
    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + ' T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + ' G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + ' M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + ' K'
    return str(value)

def convert_bytes(num):
    """ This function will convert bytes to MB, GB """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0