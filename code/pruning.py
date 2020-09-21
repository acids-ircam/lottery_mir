# -*- coding: utf-8 -*-

"""
####################

# Pruning strategies

# Here we make heavy use of the object-oriented design pattern Strategy.
    - Pruning is a strategy that contains two sub-strategies
    
# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import re

from information.compute_estimators import compute_mutual_information
from models.wavenet.wavenet import Conv1d

"""
###################

Pruning strategy (abstract class)

###################
"""
class PruningStrategy(nn.Module):
    
    def __init__(self, args):
        super(PruningStrategy, self).__init__()
        self.selection_strategy = None
        self.reset_strategy = None
        self.leaf_modules = []
        self.norm_modules = []
        self.prune_parameters = []
        self.percent = args.prune_percent
        self.initial_state_dict = None
        self.rewind_state_dict = None
        self.initializer = args.initializer
        self.prune_reset = args.prune_reset
        self.prune_scope = args.prune_scope
        self.prune_scale = args.prune_scale
        self.prune_selection = args.prune_selection
        self.rewind_it = args.rewind_it
        self.args = args
        
    def initialize(self, model):
        """ Initialize the pruning strategy """
        pass
        
    def step(self, model):
        """ One step of pruning after training """
        pass
    
    def reset(self, model):
        """ Reset the network to some given values """
        self.reset_strategy(model)
        
    def retrieve_leaf_modules(self, model):
        """ Retrieve only the leaf modules (those we might act on) """
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                self.leaf_modules.append((name, m))
        return self.leaf_modules
        
    def retrieve_prune_modules(self, model):
        """ Retrieve only prunable modules (those we might act on) """
        if (len(self.leaf_modules) == 0):
            self.retrieve_leaf_modules(model)
        self.prune_modules = []
        for name, m in self.leaf_modules:
            # Skip non-prunable layers
            if (hasattr(m, 'unprunable') and m.unprunable):
                continue
            if (m.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, Conv1d]):
                self.prune_modules.append(m)
            if (m.__class__ in [nn.RNN, nn.GRU, nn.GRUCell, nn.LSTM]):
                self.prune_modules.append(m)
        return self.prune_modules
    
    def associate_normalization_layers(self, model):
        """ 
            Associate normalization layers to their preceding weighted layer
            for eventual later pruning action 
        """
        if (len(self.leaf_modules) == 0):
            self.retrieve_leaf_modules(model)    
        # Association list
        self.norm_modules = []
        self.prune_modules = []
        # Current weighted layer
        cur_weighted = None
        # Associate norm layers to their immediate previous weighted layers
        for name, m in self.leaf_modules:
            if (m.__class__ in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]):
                cur_weighted = m
            if (m.__class__ in [nn.RNN, nn.GRU, nn.GRUCell, nn.LSTM]):
                cur_weighted = m
            if ('Norm' in str(m.__class__)):
                if (cur_weighted is not None):
                    self.norm_modules.append((m, cur_weighted))
    
    def set_pruning_parameters(self, model):
        """ Retrieve the set of weight matrices to prune """
        if (len(self.leaf_modules) == 0):
            self.retrieve_leaf_modules(model)
        self.prune_parameters = []
        for name, m in self.leaf_modules:
            # Skip non-prunable layers
            if (hasattr(m, 'unprunable') and m.unprunable):
                continue
            # Skip normalization layers parameters
            if ('Norm' in str(m.__class__)):
                continue
            if (hasattr(m, 'weight')):
                self.prune_parameters.append((name, m.weight))
            else:
                # Collect all RNN parameters starting with weight
                for n in m.__dir__():
                    if (re.search('^weight*', n) is not None):
                        self.prune_parameters.append((name + '_'  + n, getattr(m, n)))
                        
                
    def save_pruning_parameters(self, model):
        """ Save the current values of weights for later pruning """
        if (len(self.prune_parameters) == 0):
            self.set_pruning_parameters(model)
        self.parameters_snapshot = []
        for name, m in self.parameters:
            copy_weights = m.weight.data.copy()
            self.parameters_snapshot.append((name, copy_weights))
    
    def train_callback(self, model):
        """ Callback in the middle of training (before backprop) """
        pass
    
    def set_selection_strategy(self, strategy):
        self.selection_strategy = strategy
    
    def set_reset_strategy(self, strategy):
        self.reset_strategy = strategy

"""
###################

Selection strategy (abstract class)

###################
"""
class SelectionStrategy(nn.Module):
    
    def __init__(self):
        super(PruningStrategy, self).__init__()
        self.type = 'local'
        self.reset_strategy = None
    
    def set_selection_strategy(self, strategy):
        self.selection_strategy = strategy
    
    def set_reset_strategy(self, strategy):
        self.reset_strategy = strategy

"""
###################

Resetting strategy (abstract class)

###################
"""
class ResettingStrategy(nn.Module):
    
    def __init__(self):
        super(ResettingStrategy, self).__init__()
        self.selection_strategy = None
        self.reset_strategy = None
        
    def step(self, model):
        """ One step of pruning after training """
        pass
    
    def reset(self, model):
        self.reset_strategy(model)
    
    def train_callback(self, model):
        """ Callback in the middle of training (before backprop) """
        pass
    
    def set_selection_strategy(self, strategy):
        self.selection_strategy = strategy
    
    def set_reset_strategy(self, strategy):
        self.reset_strategy = strategy

"""
###################

Utilities function for pruning strategies

###################
""" 

def append_hook(module, input, output):
    dim_limit = int(1e3)
    if (not hasattr(module, 'outputs') or module.outputs is None):
        module.inputs = []
        module.outputs = []
    if (type(input) == tuple):
        input = input[0]
    if (type(output) == tuple):
        output = output[0]
        output = output.transpose(1, 2)
        input = input.transpose(1, 2)
    if (module.__class__ in [nn.Linear] and len(output.shape) > 2):
        output = output.transpose(1, 2)
    input = input.reshape(input.shape[0], -1)
    if (len(output.shape) > 2):
        output = output.reshape(output.shape[0], output.shape[1], -1)
    if (input.shape[1] > dim_limit):
        idx = np.linspace(0, input.shape[1] - 1, input.shape[1], dtype=np.int32)
        np.random.shuffle(idx)
        input = input[:, idx[:dim_limit]]
    if (len(output.shape) > 2 and output.shape[2] > (dim_limit / 10)):
        idx = np.linspace(0, output.shape[2] - 1, output.shape[2], dtype=np.int32)
        np.random.shuffle(idx)
        output = output[:, :, idx[:int(dim_limit / 10)]]
    module.inputs.append(input)
    module.outputs.append(output)
    
# New version to perform torch percentile
def torch_percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
    
"""
###################

Pruning strategy (masking in the weights matrix)

###################
""" 
class PruningMasking(PruningStrategy):
    
    def __init__(self, args):
        super(PruningMasking, self).__init__(args)
        self.mask = None
        self.mask_stats = []
    
    # Function to make an empty mask of the same size as the model
    def initialize(self, model):
        """ Initialize the pruning with masking """
        # Retrieve all parameters on which to act
        self.set_pruning_parameters(model)
        # Create a set of masks for each layer
        mask = [None] * len(self.prune_parameters)
        for step, (name, param) in enumerate(self.prune_parameters):
            mask[step] = torch.ones_like(param.data).detach()#.cpu().numpy()
        # Save mask
        self.mask = mask
        # Save the current model weights
        self.initial_state_dict = None
    
    def train_callback(self, model, epoch):
        """ Callback during training to block the gradient """
        # Check if we are at the targeted epoch to save weights
        if (self.rewind_it == epoch and self.rewind_state_dict is None):
            # Save the current model weights
            self.rewind_state_dict = copy.deepcopy(model.state_dict()) 
        # Block the gradient for masked weights
        for step, (name, param) in enumerate(self.prune_parameters):
            try:
                grad_tensor = param.grad.data #.cpu().numpy()
            except Exception as e:
                print(str(e))
                print(step, name)
                exit()
            grad_tensor = torch.where(self.mask[step] < self.args.eps, torch.Tensor([0.]).to(grad_tensor.device, non_blocking=True), grad_tensor) #np.where(self.mask[step] < self.args.eps, 0, grad_tensor)
            param.grad.data = grad_tensor #torch.from_numpy(grad_tensor).to(self.args.device)
    
    def ranking(self, name, weights, percentile_value, cur_mask):
        if (self.prune_selection in ['magnitude', 'batchnorm', 'gradient_min', 'gradient_max']):
            # Compute the new mask
            new_mask = torch.where(abs(weights) < percentile_value, torch.Tensor([0.]).to(cur_mask.device, non_blocking=True), cur_mask) #np.where(abs(weights) < percentile_value, 0, cur_mask)
        if (self.prune_selection == 'increase'):
            value = abs(abs(self.rewind_state_dict[name]) - abs(full_weights))
            # Compute global percentile
            percentile_value = torch_percentile(value, self.percent) #np.percentile(value, self.percent)
            # Compute the new mask
            new_mask = torch.where(value < percentile_value, torch.Tensor([0.]).to(cur_mask.device, non_blocking=True), cur_mask) #np.where(value < percentile_value, 0, cur_mask)
        return new_mask
    
    # Prune by Percentile module
    def step(self, model):
        """ Compute the threshold and apply the mask """
        weights = []
        if (self.prune_scope == 'global'):
            # First collect all weights
            for step, (name, param) in enumerate(self.prune_parameters):
                scale = 1
                # Pointer to the original tensor
                tensor = param.data#.cpu().numpy()
                # Gradient-based selection
                if (self.prune_selection == 'gradient_max'):
                    grad = param.grad#.cpu().numpy()
                    tensor = grad[torch.nonzero(tensor, as_tuple=True)]#np.nonzero(tensor)]
                elif (self.prune_selection == 'gradient_min'):
                    grad = 1 / (torch.abs(param.grad) + 1e-7) #.cpu().numpy()
                    tensor = grad[torch.nonzero(tensor, as_tuple=True)]#np.nonzero(tensor)]
                    #tensor = torch.abs(torch.max(tensor) - tensor) #np.max(tensor) - tensor
                # Retrieve non-pruned weights
                if (self.prune_scale == 'dimension'):
                    scale = tensor.size
                if (self.prune_scale == 'normalize'):
                    scale = torch.max(torch.abs(tensor))#np.max(np.abs(tensor))
                if (self.prune_scale == 'xavier'):
                    scale = 1.0 / np.sqrt(2.0 / tensor.shape[0] + tensor.shape[1])
                alive = tensor[torch.nonzero(tensor, as_tuple=True)]#np.nonzero(tensor)]
                alive /= scale 
                # Add to global weights
                weights.append(alive)
            # Flatten the whole weights
            weights = torch.cat(weights)#np.concatenate(weights)
            value = abs(weights)
            # Compute global percentile
            percentile_value = torch_percentile(value, self.percent) #np.percentile(value, self.percent)
        # Now apply the global or compute local factor
        for step, (name, param) in enumerate(self.prune_parameters):
            scale = 1
            # Pointer to the original tensor
            tensor = param.data #.cpu().numpy()
            # Gradient-based selection
            if (self.prune_selection == 'gradient_max'):
                tensor = param.grad#.cpu().numpy()
            elif (self.prune_selection == 'gradient_min'):
                tensor = 1.0 / (torch.abs(param.grad) + 1e-7)#np.abs(param.grad.cpu().numpy())
                #tensor = torch.abs(torch.max(tensor) - tensor) #np.max(tensor) - tensor
            # Compute scaling
            if (self.prune_scale == 'dimension'):
                scale = tensor.size
            if (self.prune_scale == 'normalize'):
                scale = torch.max(torch.abs(tensor)) #np.max(np.abs(tensor))
            if (self.prune_scale == 'xavier'):
                scale = 1.0 / np.sqrt(2.0 / tensor.shape[0] + tensor.shape[1])
            local_weights = tensor
            local_weights /= scale
            # We do not prune bias term
            if (self.prune_scope == 'local'):
                weights = tensor[torch.nonzero(param.data, as_tuple=True)] #tensor[np.nonzero(tensor)]
                # Retrieve non-pruned weights
                value = abs(weights)
                # Compute global percentile
                percentile_value = torch_percentile(value, self.percent)
            # Use the selection function to compute mask
            new_mask = PruningMasking.ranking(self, name, local_weights, percentile_value, self.mask[step])
            # Store the computed mask
            self.mask[step] = new_mask
        step = 0
        return model 
    
    def reset(self, model):
        if self.prune_reset == 'reinit':            
            # Re-init all weights
            model.apply(self.initializer)
        else:
            #model.apply(self.initializer)
            # Rewind the weights to the saved state dict
            for name, param in model.named_parameters(): 
                param.data = self.rewind_state_dict[name].clone()
        # Apply mask through the layers
        for step, (name, param) in enumerate(self.prune_parameters):
            #weight_dev = param.device
            param.data = param.data * self.mask[step] #torch.from_numpy(param.data.cpu().numpy() * self.mask[step]).to(weight_dev)
        #self.rewind_state_dict = copy.deepcopy(model.state_dict())
        self.print_masking_results()
        return model
    
    def print_masking_results(self):
        # Go through all masked parameters
        cur_stats = [0, 0, 0, 0]
        for step, (name, param) in enumerate(self.prune_parameters):
            mask_vals = self.mask[step] #torch.from_numpy(self.mask[step])
            ax_dims = list(torch.arange(1, len(self.mask[step].shape)))
            values_off = torch.sum(mask_vals == 0, ax_dims)
            values_on = torch.sum(mask_vals != 0, ax_dims)
            prunable = torch.sum(values_on == 0)
            cur_stats[0] += sum(values_off)
            cur_stats[1] += mask_vals.nelement()
            cur_stats[2] += prunable
            cur_stats[3] += len(values_on)
        self.mask_stats.append(cur_stats)

"""
###################

Pruning strategy by trimming (physically remove pruned weights)

###################
""" 

class PruningTrimming(PruningStrategy):
    
    def __init__(self, args):
        super(PruningTrimming, self).__init__(args)
    
    # Function to make an empty mask of the same size as the model
    def initialize(self, model):
        # Save the current model weights
        self.initial_state_dict = copy.deepcopy(model.state_dict())
        if ('norm' in self.prune_selection):
            # List associates normalization and weight layers
            self.associate_normalization_layers(model)
        else:
            # Retrieve all potentially prunable layers
            self.retrieve_prune_modules(model)
    
    def train_callback(self, model, iteration):
        """
        Callback function called at each training iteration
        (prior to gradient updates)

        Parameters
        ----------
        model           : reference on the trained model 
        iteration       : (int) - index of current iteration

        """
        if (self.rewind_it == iteration and self.rewind_state_dict is None):
            # Save the current model weights
            self.rewind_state_dict = copy.deepcopy(model.state_dict())
            
    def register_activation_hooks(self, model):
        """
        Function to add forward hooks to the model that will keep the
        inputs and outputs for each prunable layer. The resulting structure
        will be a list of tensors stored in each layer

        Parameters
        ----------
        model           : reference on the trained model

        """
        self.outputs = []
        self.hooks = []
        b = None
        self.list_mods = self.prune_modules
        if ('norm' in self.prune_selection):
            self.list_mods = self.norm_modules
        for l in self.list_mods:
            if ('norm' in self.list_mods):
                (b, l) = l
            # Skip non-prunable layers
            if (hasattr(l, 'unprunable') and l.unprunable):
                continue
            hook_handle = l.register_forward_hook(append_hook)
            self.hooks.append(hook_handle)
    
    def remove_activation_hooks(self):
        """
        Function that removes all previously defined forward hooks from the model
        """
        for h in self.hooks:
            h.remove()
            h = None
        for l in self.list_mods:
            if ('norm' in self.list_mods):
                (b, l) = l
            # Skip non-prunable layers
            if (hasattr(l, 'prune_values')):
                l.prune_values = None
        self.hooks = None

    def reorganize_weights(self, layer, cur_w):
        """
        Reorganize weights in a recurrent layer to reflect the unit-based
        structure in LSTM and GRU.

        Parameters
        ----------
        layer       : Current layer to re-organize
        cur_w       : Weight matrix to act on

        Returns
        -------
        cur_w       : Reorganized weights
        
        """
        if (layer.__class__ == nn.LSTM):
            w_ii, w_if, w_ic, w_io = cur_w.chunk(4, 0)
            cur_w = torch.cat((w_ii, w_if, w_ic, w_io), axis=1)
        if (layer.__class__ == nn.GRU or layer.__class__ == nn.GRUCell):
            w_ii, w_if, w_ic = cur_w.chunk(3, 0)
            cur_w = torch.cat((w_ii, w_if, w_ic), axis=1)
        return cur_w
    
    def get_layer_values(self, layer, norm=None, values=None):
        """
        Function to retrieve what values we are going to use for each layer
        in order to decide which units we should prune

        Parameters
        ----------
        layer       : Current layer (nn.Module)
        norm        : Optionally give an associated norm layer (batchnorm trimming)

        Returns
        -------
        values      : Matrix containing the indicating values

        """
        scale = 1
        if (hasattr(layer, 'prune_values') and (layer.prune_values is not None)):
            values = layer.prune_values
            return values
        if (values is not None):
            if (self.prune_selection == 'magnitude'):
                values = torch.abs(values.data)#.cpu())
            elif (self.prune_selection == 'gradient_max'):
                values = torch.abs(values.grad)#.cpu())
            elif (self.prune_selection == 'gradient_min'):
                values = 1 / (torch.abs(values.grad) + 1e-7) #torch.max(torch.abs(values.grad)) - torch.abs(values.grad)#torch.max(torch.abs(values.grad.cpu())) - torch.abs(values.grad.cpu())
            matrix = self.reorganize_weights(layer, values)
        if (self.prune_selection in ['magnitude', 'gradient_min', 'gradient_max']):
            if (values is None):
                if (self.prune_selection == 'magnitude'):
                    matrix = torch.abs(layer.weight.data)#.cpu())
                elif (self.prune_selection == 'gradient_max'):
                    matrix = torch.abs(layer.weight.grad)#.cpu())
                elif (self.prune_selection == 'gradient_min'):
                    matrix = 1 / (torch.abs(layer.weight.grad) + 1e-7) #torch.max(torch.abs(layer.weight.grad)) - torch.abs(layer.weight.grad)#torch.max(torch.abs(layer.weight.grad.cpu())) - torch.abs(layer.weight.grad.cpu())
                if ('Transpose' in str(layer.__class__)):
                    matrix = matrix.transpose(0, 1)
            ax_dims = list(torch.arange(1, len(matrix.shape)))
            values = torch.sum(matrix, ax_dims)
        elif (self.prune_selection == 'activation'):
            values = layer.outputs
            values = torch.cat(values, dim=0).detach()
            # Sum all examples
            values = torch.sum(torch.abs(values), dim=0)
            # Sum all remaining dimensions (non-examples)
            if (len(values.shape) > 1):
                ax_dims = list(torch.arange(1, len(values.shape)))
                values = torch.sum(torch.abs(values), dim=ax_dims)
            layer.inputs = None
            layer.outputs = None
        elif (self.prune_selection in ['information', 'info_target']):
            values_out = torch.cat(layer.outputs, dim=0).detach()
            if (len(values_out.shape) > 2):
                values_out = values_out.reshape(values_out.shape[0], values_out.shape[1], -1)#.cpu().numpy()
            else:
                values_out = values_out.reshape(values_out.shape[0], -1)#.cpu().numpy()
            # Output normalization
            # scale = (values_out.size) / values_out.shape[0]
            values_in = layer.inputs
            if (type(layer.inputs[0]) == tuple):
                values_in = [l_in[0]  for l_in in layer.inputs]
            values_in = torch.cat(values_in, dim=0).detach()
            values_in = values_in.reshape(values_in.shape[0], -1)#.cpu().numpy()
            values = compute_mutual_information(values_in, values_out)
            layer.inputs = None
            layer.outputs = None
        elif (self.prune_selection == 'batchnorm'):
            values = torch.abs(norm.weight.data)#.cpu())
        """
        Perform scaling (only size-based as of now)
        """
        if (self.prune_scale == 'dimension' and scale == 1):
            for d in range(len(values.shape)):
                scale *= values.shape[d]
        if (self.prune_scale == 'normalize'):
            scale = torch.max(torch.abs(values))
        if (self.prune_scale == 'xavier'):
            scale = 1.0 / np.sqrt(2.0 / values.shape[0] + values.shape[1])
        values /= scale
        if (hasattr(layer, 'prune_values')):
            layer.prune_values = values
        # Save prune values for slow procedures
        if (self.prune_selection in ['activation', 'information', 'info_target']):
            layer.prune_values = values
        #print(layer)
        #print(values.shape)
        return values
            
    def select_weights(self, layer, norm=None, cutoff=None, rank=False, values=None):
        """
        Function to perform the weight selection

        Parameters
        ----------
        layer       : Current layer (nn.Module)
        norm        : Optionally give an associated norm layer (batchnorm trimming)
        cutoff      : Cutoff value computed across layers (global mode)

        Returns
        -------
        idx         : List of indexes to keep

        """
        values = self.get_layer_values(layer, norm, values)
        # Only get values
        if (rank == False):
            return values
        if (cutoff is None):
            # Local version
            idx = torch.argsort(values, descending=True)    
            # Compute the final kept IDs
            idx_kept = int((100-self.percent)/100*len(idx))
            idx = idx[:idx_kept]
        else:
            # Keep only values above cutoff
            idx = torch.where(values > cutoff)[0]#.cpu()
        return idx

    def ranking_recurrent(self, layer, norm=None, cutoff=None, rank=False):
        full_idx = []
        for l in range(32):
            if (not hasattr(layer, 'weight_ih_l' + str(l))):
                break
            cur_ih = getattr(layer, 'weight_ih_l' + str(l))
            cur_hh = getattr(layer, 'weight_hh_l' + str(l))
            idx_ih = self.select_weights(layer, norm, cutoff, rank, cur_ih)
            idx_hh = self.select_weights(layer, norm, cutoff, rank, cur_hh)
            if (rank == True):
                full_idx.append((idx_ih, idx_hh))
            else:
                full_idx.append(idx_ih)
                full_idx.append(idx_hh)
        if (rank == False):
            full_idx = torch.cat([torch.flatten(x) for x in full_idx])
            return full_idx
        layer.unprune_idx = full_idx
        if (norm is not None):
            norm.unprune_idx = idx_hh
    
    def ranking(self, layer, norm=None, cutoff=None, rank=False):
        # Compute ranking from magnitude of kernels
        idx = self.select_weights(layer, norm, cutoff, rank)
        if (rank == False):
            return idx
        # Compute the final kept IDs
        if (len(idx) < 3):
            idx = np.linspace(0, len(layer.unprune_idx) - 1, len(layer.unprune_idx)).astype('int')
        layer.unprune_idx = idx
        if (norm is not None):
            norm.unprune_idx = idx
    
    def global_cutoff(self, model):
        b = None
        self.list_mods = self.prune_modules
        full_values = []
        if ('norm' in self.prune_selection):
            self.list_mods = self.norm_modules
        # Retrieve all values
        for l in self.list_mods:
            if ('norm' in self.prune_selection):
                (b, l) = l
            # Skip non-prunable layers
            if (hasattr(l, 'unprunable') and l.unprunable):
                continue
            if (self.prune_selection == 'info_target'):
                l.inputs = model.outputs
            if (l.__class__ in [nn.RNN, nn.LSTM, nn.GRU, nn.GRUCell]):
                values = self.ranking_recurrent(l, b, cutoff=None, rank=False)
            else:
                values = self.ranking(l, b, cutoff=None, rank=False)
            full_values.append(values)
        # Flatten all values
        value = torch.cat(full_values)
        # Compute global percentile
        cutoff = torch_percentile(value, self.percent)
        return cutoff
        
    # Prune by Percentile module
    def step(self, model):
        b, cutoff = None, None
        if (self.prune_scope == 'global'):
            cutoff = self.global_cutoff(model)
        self.list_mods = self.prune_modules
        if ('norm' in self.prune_selection):
            self.list_mods = self.norm_modules
        for l in self.list_mods:
            print(l)
            if ('norm' in self.prune_selection):
                (b, l) = l
            if (self.prune_selection == 'info_target'):
                l.inputs = model.outputs
            # Skip non-prunable layers
            if (hasattr(l, 'unprunable') and l.unprunable):
                continue
            if (l.__class__ in [nn.RNN, nn.LSTM, nn.GRU, nn.GRUCell]):
                self.ranking_recurrent(l, b, cutoff, True)
            else:
                self.ranking(l, b, cutoff, True)
            # Eventually remove values
            l.prune_values = None
        return model 
    
    def reset(self, model):
        """ Reset a network by removing units and/or channels """
        def replace_parameters(module, target_weight, target_bias=None):
            module.weight = nn.Parameter(target_weight)#torch.from_numpy(target_weight).to(self.args.device))
            if (hasattr(module, 'bias')):
                module.bias = nn.Parameter(target_bias)#torch.from_numpy(target_bias).to(self.args.device))
        def replace_recurrent(module, l, cur_idx, prev_kept):
            # Retrieve parameters
            cur_ih = getattr(module, 'weight_ih_l' + str(l)).data#.cpu().numpy()
            cur_hh = getattr(module, 'weight_hh_l' + str(l)).data#.cpu().numpy()
            cur_bih = getattr(module, 'bias_ih_l' + str(l)).data#.cpu().numpy()
            cur_bhh = getattr(module, 'bias_hh_l' + str(l)).data#.cpu().numpy()
            cur_hidden = cur_hh.shape[1]
            if (prev_kept is not None): 
                cur_ih = cur_ih[:, prev_kept]
            if (len(cur_idx[0]) < 3):
                n_hid = cur_ih.shape[0] / {nn.LSTM:4, nn.GRU:3}[module.__class__]
                cur_idx[0] = np.linspace(0, n_hid - 1, n_hid).astype('int')
                setattr(module, 'weight_ih_l' + str(l), cur_ih)
                return
            if (len(cur_idx[1]) < 3):
                cur_idx[1] = np.linspace(0, cur_hh.shape[0] - 1, cur_hh.shape[0]).astype('int')
                setattr(module, 'weight_ih_l' + str(l), cur_ih)
                return
            cur_hh = cur_hh[:, cur_idx[0]]
            rep_id0, rep_id1 = cur_idx[0], cur_idx[1]
            # Handle repetitions for LSTM and GRU
            if (module.__class__ in [nn.LSTM, nn.GRU, nn.GRUCell]):
                n_reps = {nn.LSTM:4, nn.GRU:3, nn.GRUCell:3}[module.__class__]
                final_id0, final_id1 = [], []
                for i in range(n_reps):
                    final_id0.extend(rep_id0 + (cur_hidden * i))
                    final_id1.extend(rep_id1 + (cur_hidden * i))
                rep_id0, rep_id1 = final_id0, final_id1
            # Finally replace parameters
            cur_ih = nn.Parameter(cur_ih[rep_id0])#torch.from_numpy(cur_ih[rep_id0]).to(self.args.device))
            cur_hh = nn.Parameter(cur_hh[rep_id1])#torch.from_numpy(cur_hh[rep_id1]).to(self.args.device))
            cur_bih = nn.Parameter(cur_bih[rep_id0])#torch.from_numpy(cur_bih[rep_id0]).to(self.args.device))
            cur_bhh = nn.Parameter(cur_bhh[rep_id1])#torch.from_numpy(cur_bhh[rep_id1]).to(self.args.device))
            setattr(module, 'weight_ih_l' + str(l), cur_ih)
            setattr(module, 'weight_hh_l' + str(l), cur_hh)
            setattr(module, 'bias_ih_l' + str(l), cur_bih)
            setattr(module, 'bias_hh_l' + str(l), cur_bhh)
            module.hidden_size = len(cur_idx[1])
        # Possibility to reinit
        if self.prune_reset == 'reinit':        
            model.apply(self.initializer)
        else:
            model.apply(self.initializer)
            # Rewind the weights to the saved state dict
            for name, param in model.named_parameters():
                param.data = self.rewind_state_dict[name]
        # Need to track previous modules
        prev_kept = None
        for (name, m) in self.leaf_modules:
            if (hasattr(m, 'unprune_idx')):
                if ('Transpose' in str(m.__class__)):
                    kept_weights = m.weight.data[:, m.unprune_idx]#.cpu().numpy()[:, m.unprune_idx]
                    if (prev_kept is not None and len(kept_weights.shape) > 1):
                        kept_weights = kept_weights[prev_kept]
                elif (m.__class__ in [nn.LSTM, nn.GRU, nn.GRUCell, nn.RNN]):
                    for l in range(32):
                        if (not hasattr(m, 'weight_ih_l' + str(l))):
                            break
                        replace_recurrent(m, l, m.unprune_idx[l], prev_kept)
                        prev_kept = m.unprune_idx[l][1]
                    continue
                else:
                    kept_weights = m.weight.data[m.unprune_idx]#.cpu().numpy()[m.unprune_idx]
                    if (prev_kept is not None and len(kept_weights.shape) > 1):
                        kept_weights = kept_weights[:, prev_kept]
                if (hasattr(m, 'bias')):
                    kept_biases = m.bias.data[m.unprune_idx]#.cpu().numpy()[m.unprune_idx]
                replace_parameters(m, kept_weights, kept_biases)
                prev_kept = m.unprune_idx
                if ('Norm' in str(m.__class__) and (not 'LayerNorm' in str(m.__class__))):
                    running_mean = m.running_mean[m.unprune_idx]#.cpu().numpy()[m.unprune_idx]
                    m.running_mean.data = running_mean#torch.from_numpy(running_mean).to(self.args.device)
                    running_var = m.running_var[m.unprune_idx]#.cpu().numpy()[m.unprune_idx]
                    m.running_var.data = running_var#torch.from_numpy(running_var).to(self.args.device)
                if ('LayerNorm' in str(m.__class__)):
                    m.normalized_shape = (m.unprune_idx.shape[0],)
            elif (hasattr(m, 'weight') and (m.weight is not None) and prev_kept is not None):
                kept_weights = m.weight.data#.cpu().numpy()
                if (prev_kept is not None):
                    if ('Transpose' in str(m.__class__) or 'Norm' in str(m.__class__)):
                        kept_weights = kept_weights[prev_kept]
                    else:
                        kept_weights = kept_weights[:, prev_kept]
                kept_biases = m.bias.data#.cpu().numpy()        
                if ('Norm' in str(m.__class__)):
                    if (not 'LayerNorm' in str(m.__class__)):
                        running_mean = m.running_mean[prev_kept]#.cpu().numpy()[prev_kept]
                        m.running_mean.data = running_mean#torch.from_numpy(running_mean).to(self.args.device)
                        running_var = m.running_var[prev_kept]#.cpu().numpy()[prev_kept]
                        m.running_var.data = running_var#torch.from_numpy(running_var).to(self.args.device)
                    kept_biases = kept_biases[prev_kept]
                    if ('LayerNorm' in str(m.__class__)):
                        m.normalized_shape = (prev_kept.shape[0],)
                else:
                    prev_kept = None
                replace_parameters(m, kept_weights, kept_biases)
            # Skip non-prunable layers
            if (hasattr(m, 'unprunable') and m.unprunable):
                prev_kept = None
                continue
        self.rewind_state_dict = copy.deepcopy(model.state_dict())
        return model
    

"""
###################

Hybrud pruning strategy by both trimming and masking

###################
""" 

class PruningHybrid(PruningTrimming, PruningMasking):
    
    def __init__(self, args):
        super(PruningTrimming, self).__init__(args)
        PruningTrimming.__init__(self, args)
        PruningMasking.__init__(self, args)
    
    # Function to make an empty mask of the same size as the model
    def initialize(self, model):
        # Call bath initialize
        PruningTrimming.initialize(self, model)
        PruningMasking.initialize(self, model)
        # Place the masks directly inside the parameters
        for step, (name, param) in enumerate(self.prune_parameters):
            param.mask = self.mask[step]
    
    def train_callback(self, model, iteration):
        # Only masking has explicit callback
        PruningMasking.train_callback(self, model, iteration)


    """ Small note on functions only in trimming
    def select_weights(self, matrix, norm=None):
        # Select weights based on criterion
    def reorganize_weights(self, layer, cur_w):
        # Used to reorganize GRU - LSTM weights
    def ranking_recurrent(self, layer, norm=None):
        # Used to perform ranking in recurrent
    """
        
    # Prune by Percentile module
    def step(self, model):
        return PruningTrimming.step(self, model) 
    
    def switch_magnitude(self):
        self.real_selection = self.prune_selection
        self.prune_selection = 'magnitude'
        
    def unswitch_magnitude(self):
        self.prune_selection = self.real_selection
    
    def reset(self, model):
        """ Reset a network by removing units and/or channels """
        def replace_parameters(module, target_weight, target_bias=None):
            module.weight = nn.Parameter(torch.from_numpy(target_weight).to(self.args.device))
            if (hasattr(module, 'bias')):
                module.bias = nn.Parameter(torch.from_numpy(target_bias).to(self.args.device))
        def replace_recurrent(module, l, cur_idx, prev_kept):
            # Retrieve parameters
            cur_ih_m = getattr(module, 'weight_ih_l' + str(l))
            cur_ih = cur_ih_m.data.cpu().numpy()
            cur_hh_m = getattr(module, 'weight_hh_l' + str(l))
            cur_hh = cur_hh_m.data.cpu().numpy()
            cur_bih = getattr(module, 'bias_ih_l' + str(l)).data.cpu().numpy()
            cur_bhh = getattr(module, 'bias_hh_l' + str(l)).data.cpu().numpy()
            cur_hidden = cur_hh.shape[1]
            if (prev_kept is not None): 
                cur_ih = cur_ih[:, prev_kept]
                cur_ih_m.mask = cur_ih_m.mask[:, prev_kept]
            cur_hh = cur_hh[:, cur_idx[0]]
            cur_hh_m.mask = cur_hh_m.mask[:, cur_idx[0]]
            rep_id0, rep_id1 = cur_idx[0], cur_idx[1]
            # Handle repetitions for LSTM and GRU
            if (module.__class__ in [nn.LSTM, nn.GRU]):
                n_reps = {nn.LSTM:4, nn.GRU:3}[module.__class__]
                final_id0, final_id1 = [], []
                for i in range(n_reps):
                    final_id0.extend(rep_id0 + (cur_hidden * i))
                    final_id1.extend(rep_id1 + (cur_hidden * i))
                rep_id0, rep_id1 = final_id0, final_id1
            # Finally replace parameters
            cur_ih = nn.Parameter(torch.from_numpy(cur_ih[rep_id0]).to(self.args.device))
            cur_hh = nn.Parameter(torch.from_numpy(cur_hh[rep_id1]).to(self.args.device))
            cur_ih_m.mask = cur_ih_m.mask[rep_id0]
            cur_hh_m.mask = cur_hh_m.mask[rep_id1]
            cur_bih = nn.Parameter(torch.from_numpy(cur_bih[rep_id0]).to(self.args.device))
            cur_bhh = nn.Parameter(torch.from_numpy(cur_bhh[rep_id1]).to(self.args.device))
            setattr(module, 'weight_ih_l' + str(l), cur_ih)
            setattr(module, 'weight_hh_l' + str(l), cur_hh)
            setattr(module, 'bias_ih_l' + str(l), cur_bih)
            setattr(module, 'bias_hh_l' + str(l), cur_bhh)
            module.hidden_size = len(cur_idx[1])
        # Possibility to reinit
        if self.prune_reset == 'reinit':            
            model.apply(self.initializer)
        else:       
            model.apply(self.initializer)
            # Rewind the weights to the saved state dict
            for name, param in model.named_parameters(): 
                param.data = self.rewind_state_dict[name]
        # Need to track previous modules
        prev_kept = None
        for (name, m) in self.leaf_modules:
            if (hasattr(m, 'unprune_idx')):
                if ('Transpose' in str(m.__class__)):
                    kept_weights = m.weight.data.cpu().numpy()[:, m.unprune_idx]
                    if (hasattr(m.weight, 'mask')):
                        m.weight.mask = m.weight.mask[:, m.unprune_idx]
                    if (prev_kept is not None and len(kept_weights.shape) > 1):
                        kept_weights = kept_weights[prev_kept]
                        if (hasattr(m.weight, 'mask')):
                            m.weight.mask = m.weight.mask[prev_kept]
                elif (m.__class__ in [nn.LSTM, nn.GRU, nn.RNN]):
                    for l in range(32):
                        if (not hasattr(m, 'weight_ih_l' + str(l))):
                            break
                        replace_recurrent(m, l, m.unprune_idx[l], prev_kept)
                        prev_kept = m.unprune_idx[l][1]
                    continue
                else:
                    kept_weights = m.weight.data.cpu().numpy()[m.unprune_idx]
                    if (hasattr(m.weight, 'mask')):
                        m.weight.mask = m.weight.mask[m.unprune_idx]
                    if (prev_kept is not None and len(kept_weights.shape) > 1):
                        kept_weights = kept_weights[:, prev_kept]
                        if (hasattr(m.weight, 'mask')):
                            m.weight.mask = m.weight.mask[:, prev_kept]
                if (hasattr(m, 'bias')):
                    kept_biases = m.bias.data.cpu().numpy()[m.unprune_idx]
                replace_parameters(m, kept_weights, kept_biases)
                prev_kept = m.unprune_idx
                if ('Norm' in str(m.__class__)):
                    running_mean = m.running_mean.cpu().numpy()[m.unprune_idx]
                    m.running_mean.data = torch.from_numpy(running_mean).to(self.args.device)
                    running_var = m.running_var.cpu().numpy()[m.unprune_idx]
                    m.running_var.data = torch.from_numpy(running_var).to(self.args.device)
            elif (hasattr(m, 'weight') and (m.weight is not None) and prev_kept is not None):
                kept_weights = m.weight.data.cpu().numpy()
                if (prev_kept is not None):
                    if ('Transpose' in str(m.__class__) or 'Norm' in str(m.__class__)):
                        kept_weights = kept_weights[prev_kept]
                        if (hasattr(m.weight, 'mask')):
                            m.weight.mask = m.weight.mask[prev_kept]
                    else:
                        kept_weights = kept_weights[:, prev_kept]
                        if (hasattr(m.weight, 'mask')):
                            m.weight.mask = m.weight.mask[:, prev_kept]
                kept_biases = m.bias.data.cpu().numpy()        
                if ('Norm' in str(m.__class__)):
                    running_mean = m.running_mean.cpu().numpy()[prev_kept]
                    m.running_mean.data = torch.from_numpy(running_mean).to(self.args.device)
                    running_var = m.running_var.cpu().numpy()[prev_kept]
                    m.running_var.data = torch.from_numpy(running_var).to(self.args.device)
                    kept_biases = kept_biases[prev_kept]
                else:
                    prev_kept = None
                replace_parameters(m, kept_weights, kept_biases)
            # Skip non-prunable layers
            if (hasattr(m, 'unprunable') and m.unprunable):
                prev_kept = None
                continue
        self.switch_magnitude()
        # Replace mask by the trimmed ones
        tmp_mask = {}
        for step, (name, param) in enumerate(self.prune_parameters):
            tmp_mask[step] = param.mask
        # Reupdate the prune parameters list
        self.set_pruning_parameters(model)
        for step, (name, param) in enumerate(self.prune_parameters):
            self.mask[step] = tmp_mask[step]
            param.mask = tmp_mask[step]
        # Call the masking step and reset function
        PruningMasking.step(self, model)
        # Apply mask through the layers
        for step, (name, param) in enumerate(self.prune_parameters):
            weight_dev = param.device
            param.data = torch.from_numpy(param.data.cpu().numpy() * self.mask[step]).to(weight_dev)
        # Reset dictionnary
        self.rewind_state_dict = copy.deepcopy(model.state_dict())
        self.unswitch_magnitude()
        return model
    
