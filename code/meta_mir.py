#%% -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%%
"""
######
Parameters sets for the analysis
######
"""
# Number of training epochs
nb_epochs = 200
# Set of datasets
datasets = ['drums', 'nsynth-10000', 'instrument', 'mnist-sub', 'onsets', 'singing']
# Models grid arguments
model = ['transcribe', 'classify', 'cnnAce', 'crepe-medium', 'crepe-tiny']
# Types of sub-layers in the *AE architectures
type_mod = ['mlp', 'gated_mlp', 'cnn', 'res_cnn', 'gated_cnn', 'sing-ae']
# Pruning process arguments
initialize = ['classic', 'uniform', 'normal', 'xavier', 'kaiming']
# Type of pruning operations
pruning = ['trimming', 'masking', 'hybrid']
# Selection criteria
selection = ['magnitude', 'batchnorm', 'gradient-min', 'gradient-max', 'activation', 'information', 'info-target']
# Type of reset operation
reset = ['reinit', 'rewind']
# Scope of selection
scope = ['local', 'global']
# Scope of selection
run = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Statistics of the models
stats_names = ['parameters', 'disk_size', 'flops', 'memory_read', 'memory_write', 'prune_ratio']
stats_names += ['train_full', 'train_loss', 'valid_full', 'valid_loss', 'test_full', 'test_loss']
stats_names += ['stop_epoch', 'train_time']
stats_names += ['start_size', 'end_size', 'start_time', 'end_time']
# Keys in the models names
keys = ['dataset', 'model', 'type_mod', 'initialize', 'pruning', 'selection', 'reset', 'scope', 'run']
# Set of replacement expressions
replace_mods = {'mnist_sub':'mnist-sub', 'res_cnn':'res-cnn', 'gated_cnn':'gated-cnn', 'real_nvp':'real-nvp', 
                'gradient_min':'gradient-min', 'gradient_max':'gradient-max', 'info_target':'info-target',
                'sing_ae':'sing-ae'}
# Create a hashmap of all used parameters
param_hash = {'dataset': datasets,
             'model': model,
             'type_mod':type_mod,
             'initialize':initialize,
             'pruning':pruning,
             'selection':selection, 
             'reset':reset,
             'scope':scope,
             'run':run}
# Set of plot names
plot_names = list(param_hash.keys())
# Keep track of how many model we have
param_seen = {}
for k, v in param_hash.items():
    for val in v:
        param_seen[k + '_' + str(val)] = 0

"""
######
Simple processing operations definition
######
"""
# Check if model is in our filters
def check_name_filters(name):
    check_in = sum([r in results_list[l] for r in remove_from])
    check_out = (sum([(not (r in results_list[l])) for r in restrict_to]))
    return check_in or check_out

# Returns a dictionnary
def split_names(cur_name):
    # Refactor name
    for k, v in replace_mods.items():
        cur_name = cur_name.replace(k, v)
    key_list = cur_name.split('_')
    fields = {}
    for l in range(len(key_list)):
        fields[keys[l]] = key_list[l]
    # Retrieve metadata indexes
    meta_list = [None] * len(plot_names)
    for i in range(len(key_list)):
        param = keys[i]
        pID = plot_names.index(param)
        value = type(param_hash[param][0])(key_list[i])
        meta_list[pID] = param_hash[param].index(value)
        param_key = param + '_' + str(value)
        param_seen[param_key] += 1
    return cur_name, fields, meta_list

"""
######
Model analysis function
######
"""
def analyze_models(analyzer, it):
    # Results structure
    res_struct = {}
    # Replace model names
    split_name = analyzer.model_checkpoints[it]['initial'].split('/')
    start_path = base_results + '/' + split_name[-2] + '/' + split_name[-1]
    split_name = analyzer.model_checkpoints[it]['final'].split('/')
    end_path = base_results + '/' + split_name[-2] + '/' + split_name[-1]
    # Load start and end model
    start_model = torch.load(start_path, map_location='cpu')
    end_model = torch.load(end_path, map_location='cpu')
    # Sizes of initial and final models
    res_struct['start_size'] = os.stat(start_path)
    res_struct['end_size'] = os.stat(end_path)
    # Create fake batch of data (of size 1)
    fake_data = torch.ones(analyzer.input_size).unsqueeze(0)
    # Compute inference time for start model
    start_inference_time = time.time()
    start_model(fake_data)
    res_struct['start_inference_time'] = (time.time() - start_inference_time)
    # Compute inference time for end model
    end_inference_time = time.time()
    end_model(fake_data)
    res_struct['end_inference_time'] = (time.time() - end_inference_time)
    return res_struct

"""
######
Main analysis function
######
"""

# Cutoff by task
cut_len = {'drums':78, 'instrument':90, 'mnist-sub':17, 'nsynth-10000':100, 'onsets':33, 'singing':69}
# Extract the final results
def extract_results(analyzer, load_models=False, properties=None, name=None):
    # Create results structure
    final_stats = {}
    for n in stats_names:
        final_stats[n] = np.zeros(analyzer.prune_it)
    # Stop iterations
    final_stats['stop_epoch'][:] = analyzer.stopped_epoch
    # Training time
    final_stats['train_time'][:] = analyzer.training_times
    # Full losses curves
    final_stats['train_full'] = np.zeros((analyzer.prune_it, nb_epochs))
    final_stats['valid_full'] = np.zeros((analyzer.prune_it, nb_epochs))
    final_stats['test_full'] = np.zeros((analyzer.prune_it, nb_epochs))
    final_stats['test_full'] = np.zeros((analyzer.prune_it, nb_epochs))
    final_stats['true_prune'] = np.zeros((analyzer.prune_it, 1))
    # Perform iteration-wise analysis
    for it in range(analyzer.prune_it):
        # Saved checkpoints
        #if (len(analyzer.model_checkpoints[it]['final']) == 0):
        #    print(analyzer.model_checkpoints[it]['final'])
        #    print('Empty model results !')
        #    break
        # Eventually anlayze the models
        if (load_models):
            res_struct = analyze_models(analyzer, it)
            # Store stats
            for n in res_struct.keys():
                final_stats[n][it] = res_struct[n]
        # Structure of the pruning
        # cur_pruning = analyzer.pruning_structure[it]
        # Layer-wise memory statistics
        # cur_layer = analyzer.layers_data[it]
        # Model memory / ops usage
        cur_stats = analyzer.model_stats[it]
        if (cur_stats is None):
            print('!'*32)
            print('No stats available')
            print(analyzer.model_stats)
            print(it)
            print('!'*32)
            #x = input("Press Enter to continue...")   
            for vals in ['train_full', 'valid_full', 'test_full', 'train_loss', 'valid_loss', 'test_loss', 'prune_ratio']:
                final_stats[vals] = final_stats[vals][:it]   
            if (it < 10):
                return None
            break
            #return None
        # Keep number of parameters
        cur_params = cur_stats['parameters']
        if (it == 0):
            ref_params = cur_params
        # Statistics for pruning ratio
        final_stats['prune_ratio'][it] = float(ref_params) / float(cur_params) 
        #print(cur_params)
        if (it > 0 and final_stats['prune_ratio'][it] == 1):
            final_stats['prune_ratio'][it] = 1.0 / (.7 ** it)
        # Store stats
        for n in cur_stats.keys():
            final_stats[n][it] = cur_stats[n]
        # Losses curves
        cur_loss = analyzer.losses[it]
        """
        if (it == 5):
            plt.figure()
            print(cur_loss)
            plt.plot(cur_loss[:, 0].detach())
            plt.title('Train')
            plt.figure()
            print(cur_loss)
            plt.plot(cur_loss[:, 2].detach())
            plt.title('Test')
        """
        max_l = cur_loss.shape[0]
        #if (it < 5):
        #    print(cur_loss[:, 2])
        final_stats['train_full'][it, :max_l] = cur_loss[:200, 0].detach()
        final_stats['valid_full'][it, :max_l] = cur_loss[:200, 1].detach()
        final_stats['test_full'][it, :max_l] = cur_loss[:200, 2].detach()
        if (max_l < nb_epochs):
            final_stats['train_full'][it, max_l:] = cur_loss[-1, 0].detach()
            final_stats['valid_full'][it, max_l:] = cur_loss[-1, 1].detach()
            final_stats['test_full'][it, max_l:] = cur_loss[-1, 2].detach()
        # Remove untrained epochs (zero loss)
        cur_loss = cur_loss[cur_loss.sum(dim=1) != 0]
        # Find best test loss (based on valid)
        idx = torch.argmin(cur_loss[:, 2])
        final_stats['train_loss'][it] = cur_loss[idx, 0].detach()
        final_stats['valid_loss'][it] = cur_loss[idx, 1].detach()
        final_stats['test_loss'][it] = cur_loss[idx, 2].detach()
    print(final_stats['test_loss'])
    for vals in ['train_loss', 'valid_loss', 'test_loss']:
        for v in range(len(final_stats[vals]) - 1):
            if (final_stats[vals][v] > 2 * final_stats[vals][v + 1] or np.isnan(final_stats[vals][v])):
                if (v > 0):
                    final_stats[vals][v] = (final_stats[vals][v - 1] + final_stats[vals][v + 1]) / 2
                else:
                    final_stats[vals][0] = final_stats[vals][1] * 1.1
    print(final_stats['test_loss'])
    final_stats['test_loss_unscaled'] = final_stats['test_loss']
    if (np.isnan(np.sum(final_stats['test_loss']))):
        return None
    f = interp1d(final_stats['prune_ratio'], final_stats['test_loss'])
    x = np.linspace(1, int(final_stats['prune_ratio'][-1]), int(final_stats['prune_ratio'][-1]))
    final_stats['test_loss'] = f(x)
    final_stats['test_loss_stretch'] = f(x)
    for vals in ['train_full', 'valid_full', 'test_full', 'train_loss', 'valid_loss', 'test_loss']:
        final_stats[vals] /= final_stats[vals][0]
    plot_cols = {'masking':'r', 'trimming':'g'}
    plot_style = {'local':'--', 'global':'-'}
    #plt.plot(final_stats['prune_ratio'], final_stats['test_loss'], color=plot_cols[properties['pruning']], linestyle=plot_style[properties['scope']])
    #print('*'*32)
    #print('*'*32)
    #print('Reality check')
    print(final_stats['prune_ratio'])
    print(final_stats['test_loss'])
    print(len(final_stats['test_loss']))
    x = input("Press Enter to continue...")
    #if (x == 'r'):
    #    os.system('rm ' + name)
    #    return None
    if (x == 's'):
        return None
    if (len(final_stats['test_loss']) < cut_len[properties['dataset']]):
        return None
    final_stats['test_loss'] = final_stats['test_loss'][:cut_len[properties['dataset']]]
    return final_stats

"""
######
First parse all results and retrieve properties
######
"""
base_results = 'output/mir_final_30'
full_results = {}
# Set that models are forced to include 
restrict_to = []#'nsynth']#, 'trimming', 'classic']
# Remove the following
remove_from = ['vae']
# All results obtained
results_list = [name for name in os.listdir(base_results) if (os.path.splitext(name)[1] == '.th')]
results_list.sort()
# Finally launch the analysis
plt.figure(figsize=(12,12))
# List of kept results
kept_names = []
kept_metadata = []
kept_results = []
# Create full structure
full_results = {}
for n in stats_names:
    full_results[n] = []
plt.figure()
plt.ylim(0, 2)
task_results = {}
task_props = {}
for t in ['instrument', 'singing', 'drums', 'onsets', 'nsynth-10000', 'mnist-sub']:
    task_results[t] = []
    task_props[t] = []
# Retrieve all results
for l in range(len(results_list)):
    cur_res = base_results + '/' + results_list[l]
    print(results_list[l])
    # Retrieve and load analyzer object
    analyzer = torch.load(cur_res, map_location='cpu')
    # Split names properties
    final_name, properties, metadata = split_names(os.path.splitext(results_list[l])[0])
    print(properties)
    # Check if we include the model
    if (check_name_filters(final_name)):
        continue
    print(final_name)
    print(metadata)
    # Analyze and retrieve results
    cur_results = extract_results(analyzer, load_models=False, properties=properties, name = cur_res)
    # Check for shitty results 
    if (cur_results is None):
        print('*'*32)
        print('*'*32)
        print('Divergence removed from analysis !')
        print('*'*32)
        print('*'*32)
        continue
    # Append all results
    kept_results.append(cur_results)
    # Append to task-specific
    task_results[properties['dataset']].append(cur_results)
    task_props[properties['dataset']].append(properties)
    # Add name to the analyzed results
    kept_names.append(final_name)
    # Add metadata to the set
    kept_metadata.append(metadata)  
    # Append in separate arrays
    for n in stats_names:
        full_results[n].append(cur_results[n])
# Now transform all results into simple array
#for n in stats_names:
#    full_results[n] = np.stack(full_results[n])
# Also transform metadata
kept_metadata = np.stack(kept_metadata)
kept_names = np.array(kept_names)

#%%

def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)

def round_size(value):
    val = np.round(value * 10) / 10
    return str(val) + 'M'

crit_curves = [[], [], []]
cross_curves = [[], [], []]
# Perform task-specific analysis
for t in ['instrument', 'singing', 'nsynth-10000', 'mnist-sub', 'drums', 'onsets']:
    cur_results = task_results[t]
    cur_props = task_props[t]
    c = cur_results[0]
    print('*'*32)
    print(t)
    # Properties of original model
    print(round_value(c['parameters'][0]))
    print(round_size(c['disk_size'][0]))
    print(round_value(c['flops'][0]))
    print(round_value(c['memory_read'][0] + c['memory_write'][0]))
    print('-'*16)
    # Compute mean error
    mean_err = 0.0
    for c in cur_results:
        mean_err += c['test_loss_unscaled'][0]
    mean_err /= len(cur_results)
    print('Mean error : ' + str(mean_err))
    # Compute mean trimming error
    mean_t_err = 0.0
    nb_err = 0
    for c in range(len(cur_results)):
        prop = cur_props[c]
        if (prop['pruning'] != 'trimming'):
            continue
        res = cur_results[c]
        mean_t_err += res['test_loss_unscaled'][0]
        nb_err += 1
    #mean_t_err /= nb_err
    print('Mean trim error : ' + str(mean_t_err))
    print('-'*16)
    thresh = mean_err * 1.1
    min_curves, max_curves, mean_curves = [[]] * 3, [[]] * 3, [[]] * 3
    cur_curves = 0
    for (prune, reset) in (('trimming', 'rewind'), ('masking', 'rewind'), ('trimming', 'reinit')):
        cur_mins = [1.1, 1, 1]
        cur_bests = [None] * 3
        cur_bests_idx = [0, 0, 0]
        print('!'*8)
        print(prune)
        print(reset)
        print('!'*8)
        var_curves = []
        # Find best in trimming
        for c in range(len(cur_results)):
            prop = cur_props[c]
            if (prop['pruning'] != prune or prop['reset'] != reset):
                continue
            res = cur_results[c]
            tens = (torch.Tensor(res['test_loss_unscaled']))
            id_s = np.linspace(0, len(tens) - 1, len(tens))
            vals = id_s[tens <= mean_err]
            if (len(vals) == 0):
                vals = [0]
            best = int(vals[-1])
            vals = id_s[tens < mean_err * 1.1]
            if (len(vals) == 0):
                vals = [0]
            optim = int(vals[-1])
            vals = id_s[tens < mean_err * 1.6]
            small = int(vals[-1])
            best_v, best_idx = torch.min(tens[1:], 0)
            var_curves.append(res['test_loss'])
            if (len(res['test_loss']) > 50):
                print(res['test_loss'])
                crit_curves[cur_curves].append(np.copy(res['test_loss'][:50]))
                if (cur_curves == 0 and prop['selection'] == 'magnitude'):
                    print(prop)
                    cross_curves[0].append(res['test_loss'][:50])
                if (cur_curves == 0 and prop['selection'] == 'activation'):
                    cross_curves[1].append(res['test_loss'][:50])
                if (cur_curves == 0 and prop['selection'] == 'batchnorm'):
                    cross_curves[2].append(res['test_loss'][:50])
            if (prune == 'trimming'):
                ratio_best = res['parameters'][best] / res['parameters'][0]
                ratio_optim = res['parameters'][optim] / res['parameters'][0]
                ratio_small = res['parameters'][small] / res['parameters'][0]
            else:
                ratio_best = 1 / res['prune_ratio'][best]
                ratio_optim = 1 / res['prune_ratio'][optim]
                ratio_small = 1 / res['prune_ratio'][small]
            if (ratio_best) < cur_mins[0]:
                cur_mins[0] = ratio_best
                cur_bests[0] = res
                cur_bests_idx[0] = best
            if (ratio_optim) < cur_mins[1]:
                cur_mins[1] = ratio_optim
                cur_bests[1] = res
                cur_bests_idx[1] = optim
            if (ratio_small) < cur_mins[2]:
                cur_mins[2] = ratio_small
                cur_bests[2] = res
                cur_bests_idx[2] = small
        res_names = ['Best', 'Optim', 'Small']
        for r in range(len(res_names)):
            print('-'*16)
            print(res_names[r] + ' : ')
            if (cur_bests[r] is None):
                print('NONE')
                continue
            print(cur_bests[r]['test_loss_unscaled'][cur_bests_idx[r]])
            print(cur_mins[r])
            p_str = (round_value(cur_bests[r]['parameters'][cur_bests_idx[r]]))
            d_str = (round_size(cur_bests[r]['disk_size'][cur_bests_idx[r]]))
            f_str = (round_value(cur_bests[r]['flops'][cur_bests_idx[r]]))
            m_str = (round_value(cur_bests[r]['memory_read'][cur_bests_idx[r]] + cur_bests[r]['memory_write'][cur_bests_idx[r]]))
            print(p_str + ' - ' + d_str + ' - ' + f_str + ' - '+ m_str)
        if (len(var_curves) == 0):
            var_curves = [[0, 0], [0, 0]]
        min_curves[cur_curves] = np.min(var_curves, axis=0)
        max_curves[cur_curves] = np.max(var_curves, axis=0)
        mean_curves[cur_curves] = np.mean(var_curves, axis=0)
        cur_curves += 1
    plt.figure()
    cmap = plt.cm.get_cmap('magma', 5)
    for v in range(3):
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), mean_curves[v], c=cmap(v+1), linewidth = 2.5, alpha = 0.85)
    for v in range(3):
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), min_curves[v], c=cmap(v+1), linewidth = 0.75, alpha = 0.6)
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), max_curves[v], c=cmap(v+1), linewidth = 0.75, alpha = 0.6)
        for i in range(1, len(min_curves[v])):
            plt.fill([i, i, i+1, i+1], [min_curves[v][i-1], max_curves[v][i-1], max_curves[v][i], min_curves[v][i]], c=cmap(v+1), alpha=0.2, edgecolor=None, linewidth=0)
    plt.legend(['Trim', 'Mask', 'Prune'])
    plt.xscale('log')
    plt.title(t)
    plt.savefig('output/figures/' + t + '_curves.pdf')
    plt.close()
#%%
var_curves = [cross_curves, crit_curves]
var_names = ['cross', 'criteria']
for c in range(2):
    for v in range(3):
        min_curves[v] = np.min(var_curves[c][v], axis=0)
        max_curves[v] = np.max(var_curves[c][v], axis=0)
        mean_curves[v] = np.mean(var_curves[c][v], axis=0)
    plt.figure()
    cmap = plt.cm.get_cmap('seismic', 4)
    for v in range(3):
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), mean_curves[v], c=cmap(v+1), linewidth = 2.5, alpha = 0.85)
    for v in range(3):
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), min_curves[v], c=cmap(v+1), linewidth = 0.75, alpha = 0.6)
        plt.plot(np.linspace(1, len(mean_curves[v]), len(mean_curves[v])), max_curves[v], c=cmap(v+1), linewidth = 0.75, alpha = 0.6)
        for i in range(1, len(min_curves[v])):
            plt.fill([i, i, i+1, i+1], [min_curves[v][i-1], max_curves[v][i-1], max_curves[v][i], min_curves[v][i]], c=cmap(v+1), alpha=0.2, edgecolor=None, linewidth=0)
    plt.legend(['Magnitude', 'Activation', 'Batchnorm'])
    plt.xscale('log')
    plt.title(t)
    plt.savefig('output/figures/' + var_names[c] + '_curves.pdf')
    plt.close()

#%%
"""
######
Parse all curves to produce figures
(This mixes all kind of information, models, scope and criteria)
######
"""
plots = [datasets, model, type_mod, initialize, pruning, selection, reset, scope]
plot_name = ['dataset','model','type_mod','initialize','pruning','selection','reset','scope']
# Variants to exclude from analysis
exclude_variants = ['dataset', 'type_mod', 'initialize']
# Now analyze everything
for p in range(len(plot_name)):
    print(' - Analyzing ' + plot_name[p])
    cur_variants = plots[p];
    if (plot_name[p] in exclude_variants):
        continue
    # Now find all cross-parameters configurations
    kept_meta_vars = kept_metadata.copy()
    kept_meta_vars[:, p] = 0
    meta_identical = np.zeros((kept_metadata.shape[0], kept_metadata.shape[0]))
    for i in range(kept_metadata.shape[0]):
        for j in range(i + 1, kept_metadata.shape[0]):
            if (np.sum(np.abs(kept_meta_vars[i, :] - kept_meta_vars[j, :])) == 0):
                meta_identical[i, j] = 1
    axis_legend = []
    N = len(cur_variants)
    cmap = plt.cm.get_cmap('jet', len(cur_variants))
    cur_test_curve = full_results['test_loss']
    # Fill all configurations
    mean_curves, min_curves, max_curves, max_stats = [None] * N, [None] * N, [None] * N, []
    mean_curves_var, min_curves_var, max_curves_var, max_stats_var = [None] * N, [None] * N, [None] * N, [None] * N
    axis_legend_var = [None] * N
    best_names, best_small_names = [None] * N, [None] * N
    for v in range(len(cur_variants)):
        print('   . Variant ' + str(cur_variants[v]))
        var_curves = cur_test_curve[kept_metadata[:, p] == v]
        names = kept_names[kept_metadata[:, p] == v]
        print(names)
        if (len(var_curves) == 0):
            var_curves = np.zeros((2, 2))
            names = ['none']
        # Compute curves
        min_curves[v] = np.min(var_curves, axis=0)
        max_curves[v] = np.max(var_curves, axis=0)
        mean_curves[v] = np.mean(var_curves, axis=0)
        # Find best (across all pruning)
        best_idx = np.argmin(np.min(var_curves, axis=1))
        best_names[v] = (names[best_idx], np.min(var_curves, axis=1)[best_idx])
        # Find best at maximal pruning
        best_idx = np.argmin(var_curves[:, -1])
        best_small_names[v] = (names[best_idx], var_curves[best_idx, -1])
        # Compute stats
        max_stats.append(np.min(var_curves, axis=1))
        axis_legend.append(cur_variants[v])
        # Create sub-lists
        mean_curves_var[v], min_curves_var[v], max_curves_var[v], max_stats_var[v] = [None] * len(plot_name), [None] * len(plot_name), [None] * len(plot_name), [None] * len(plot_name)
        axis_legend_var[v] = [None] * len(plot_name)
        for p2 in range(len(plot_name)):
            N2 = len(plots[p2])
            mean_curves_2, min_curves_2, max_curves_2, max_stats_2 = [None] * N2, [None] * N2, [None] * N2, []
            axis_legend_2 = []
            for v2 in range(len(plots[p2])):
                var_curves = cur_test_curve[np.logical_and((kept_metadata[:, p] == v), (kept_metadata[:, p2] == v2))]
                if (len(var_curves) == 0):
                    var_curves = np.zeros((1, 1))
                # Compute curves
                min_curves_2[v2] = np.min(var_curves, axis=0)
                max_curves_2[v2] = np.max(var_curves, axis=0)
                mean_curves_2[v2] = np.mean(var_curves, axis=0)
                # Compute stats
                max_stats_2.append(np.min(var_curves, axis=1))
                axis_legend_2.append(plots[p2][v2])
            mean_curves_var[v][p2] = mean_curves_2
            max_curves_var[v][p2] = max_curves_2
            min_curves_var[v][p2] = min_curves_2
            max_stats_var[v][p2] = max_stats_2
            axis_legend_var[v][p2] = axis_legend_2
    # Print best
    print('Best setting (across pruning)')
    print(best_names)
    print('Best and smallest setting (maximal pruning)')
    print(best_small_names)
    # Boxplot figure
    plt.figure()
    plt.boxplot(max_stats)
    plt.title(plot_name[p])
    plt.xticks(np.linspace(1, len(cur_variants), len(cur_variants)), axis_legend)
    plt.savefig('output/figures/full/results_'+plot_name[p]+'_boxplot.pdf')
    plt.close()
    # 
    plt.figure()
    for v in range(len(cur_variants)):
        plt.plot(mean_curves[v], c=cmap(v), linewidth = 2.5, alpha = 0.85)
    for v in range(len(cur_variants)):
        plt.plot(min_curves[v], c=cmap(v), linewidth = 0.75, alpha = 0.6)
        plt.plot(max_curves[v], c=cmap(v), linewidth = 0.75, alpha = 0.6)
        for i in range(1, len(min_curves[v])):
            plt.fill([i-1, i-1, i, i], [min_curves[v][i-1], max_curves[v][i-1], max_curves[v][i], min_curves[v][i]], c=cmap(v), alpha=0.2, edgecolor=None, linewidth=0)
    plt.legend(axis_legend)
    plt.xscale('log')
    plt.title(plot_name[p])
    plt.savefig('output/figures/full/results_'+plot_name[p]+'_curves.pdf')
    plt.close()
    # Now plot all cross - configurations
    for v in range(len(cur_variants)):
        for p2 in range(len(plot_name)):     
            if (plot_name[p] in exclude_variants):
                continue
            plt.figure()
            plt.boxplot(max_stats_var[v][p2])
            plt.title(plot_name[p])
            cur_vars_2 = plots[p2]
            cmap = plt.cm.get_cmap('jet', len(cur_vars_2))
            plt.xticks(np.linspace(1, len(cur_vars_2), len(cur_vars_2)), axis_legend)
            plt.savefig('output/figures/full/results_'+plot_name[p]+'_'+cur_variants[v]+'_'+plot_name[p2]+'_boxplot.pdf')
            plt.close()
            plt.figure()
            mean_curves_var[v][p2]
            for v2 in range(len(cur_vars_2)):
                plt.plot(mean_curves_var[v][p2][v2], c=cmap(v2), linewidth = 2.5, alpha = 0.85)
            for v2 in range(len(cur_vars_2)):
                plt.plot(min_curves_var[v][p2][v2], c=cmap(v2), linewidth = 0.75, alpha = 0.6)
                plt.plot(max_curves_var[v][p2][v2], c=cmap(v2), linewidth = 0.75, alpha = 0.6)
                for i in range(1, len(min_curves_var[v][p2][v2])):
                    plt.fill([i-1, i-1, i, i], [min_curves_var[v][p2][v2][i-1], max_curves_var[v][p2][v2][i-1], max_curves_var[v][p2][v2][i], min_curves_var[v][p2][v2][i]], c=cmap(v2), alpha=0.2, edgecolor=None, linewidth=0)
            plt.legend(axis_legend_var[v][p2])
            plt.title(plot_name[p])
            plt.xscale('log')
            plt.savefig('output/figures/full/results_'+plot_name[p]+'_'+cur_variants[v]+'_'+plot_name[p2]+'_curves.pdf')
            plt.close()

#%%                    
"""
######
######

Now produce some selected figures, by carefully taking some properties

######
######
"""
plots = [datasets, model, type_mod, initialize, pruning, selection, reset, scope]
plot_name = ['dataset','model','type_mod','initialize','pruning','selection','reset','scope']
def compare_sets(meta_sets, axis_legend, fig_name):
    plt.figure()
    cmap = plt.cm.get_cmap('jet', len(meta_sets))
    for v in range(len(meta_sets)):
        plt.plot([], c=cmap(v), linewidth=2.5)
    for v in range(len(meta_sets)):
        cur_set = meta_sets[v]
        select_values = []
        for name, value in cur_set.items():
            if (type(value) is not list):
                value = [value]
            cur_truth = [False]  * kept_metadata.shape[0]
            for cur_v in value:
                # Find property index
                id_p = plot_name.index(name)
                # Find current variant index
                id_v = param_hash[name].index(cur_v);
                # Truth value for selection
                cur_truth = np.logical_or(cur_truth, kept_metadata[:, id_p] == id_v)
            select_values.append(cur_truth)
        final_values = [True] * kept_metadata.shape[0]
        for v2 in select_values:
            final_values = np.logical_and(final_values, v2)
        # Retrieve curves
        var_curves = full_results['test_loss'][final_values]
        print(value)
        min_curves = np.min(var_curves, axis=0)
        max_curves = np.max(var_curves, axis=0)
        mean_curves = np.mean(var_curves, axis=0)
        x_full = np.linspace(1, 100, 100)
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        f = interp1d(x_full, min_curves)
        min_curves = f(x)
        f = interp1d(x_full, max_curves)
        max_curves = f(x)
        f = interp1d(x_full, mean_curves)
        mean_curves = f(x)
        # Do plotting
        plt.plot(x, mean_curves, c=cmap(v), linewidth = 2.5, alpha = 0.85)
        plt.plot(x, min_curves, c=cmap(v), linewidth = 0.75, alpha = 0.6)
        plt.plot(x, max_curves, c=cmap(v), linewidth = 0.75, alpha = 0.6)
        for i in range(1, len(min_curves)):
            plt.fill([x[i-1], x[i-1], x[i], x[i]], [min_curves[i-1], max_curves[i-1], max_curves[i], min_curves[i]], c=cmap(v), alpha=0.2, edgecolor=None, linewidth=0)
    plt.legend(axis_legend)
    plt.title(fig_name)
    plt.xscale('log')
    plt.savefig('output/figures/'+fig_name+'.pdf')
    plt.close()

# Masking vs. trimming figure
set_trimming = {'pruning':'trimming', 'scope':'global', 'reset':'rewind'}#, 'selection':['information', 'info-target']}
set_trimming_local = {'pruning':'trimming', 'scope':'local', 'reset':'rewind'}
set_masking = {'pruning':'masking', 'scope':'local', 'reset':'rewind', 'selection':'magnitude'}
compare_sets([set_trimming, set_masking, set_trimming_local], ['trimming', 'masking', 'trimming_local'], 'trimming_masking_rewind')
set_trimming = {'pruning':'trimming', 'scope':'global'}#, 'selection':['information', 'info-target']}
set_trimming_local = {'pruning':'trimming', 'scope':'local'}#, 'reset':'rewind'}
set_masking = {'pruning':'masking', 'selection':'magnitude'}
compare_sets([set_trimming, set_masking, set_trimming_local], ['trimming', 'masking', 'trimming_local'], 'trimming_masking')

#%%
# Local vs. global figure
set_local = {'pruning':'trimming', 'scope':'local'}#, 'selection':['information', 'info-target']}
set_global = {'pruning':'trimming', 'scope':'global'}#, 'selection':['information', 'info-target']}
compare_sets([set_local, set_global], ['local', 'global'], 'local_global')
compare_sets([set_local], ['local'], 'local_global_local')
compare_sets([set_global], ['global'], 'local_global_global')
set_local = {'pruning':'trimming', 'scope':'local', 'reset':'rewind'}#, 'selection':['information', 'info-target']}
set_global = {'pruning':'trimming', 'scope':'global', 'reset':'rewind'}#, 'selection':['information', 'info-target']}
compare_sets([set_local, set_global], ['local', 'global'], 'local_global_rewind')
compare_sets([set_local], ['local'], 'local_global_rewind_local')
compare_sets([set_global], ['global'], 'local_global_rewind_global')
# Subfigure in [Local] - Reinit vs. rewind
set_rewind = {'pruning':'trimming', 'scope':'local', 'reset':'rewind'}#, 'selection':['information', 'info-target']}
set_reinit = {'pruning':'trimming', 'scope':'local', 'reset':'reinit'}#, 'selection':['information', 'info-target']}
compare_sets([set_rewind, set_reinit], ['rewind', 'reinit'], 'rewind_reinit_local')
compare_sets([set_rewind], ['rewind'], 'rewind_reinit_local_rewind')
compare_sets([set_reinit], ['reinit'], 'rewind_reinit_local_reinit')
# Subfigure in [Global] - Reinit vs. rewind
set_rewind = {'pruning':'trimming', 'scope':'global', 'reset':'rewind'}#, 'selection':['information', 'info-target']}
set_reinit = {'scope':'global', 'reset':'reinit'}#, 'selection':['information', 'info-target']}
compare_sets([set_rewind, set_reinit], ['rewind', 'reinit'], 'rewind_reinit_global')
compare_sets([set_rewind], ['rewind'], 'rewind_reinit_global_rewind')
compare_sets([set_reinit], ['reinit'], 'rewind_reinit_global_reinit')
#%%
print('PUEAPEORUAP')
# Compare selection criteria (global)
set_criteria_mag = {'pruning':'trimming', 'selection':'magnitude', 'reset':'rewind'}
set_criteria_act = {'pruning':'trimming', 'selection':'activation', 'reset':'rewind'}
set_criteria_gra = {'pruning':'trimming', 'selection':'gradient-min', 'reset':'rewind'}
set_criteria_bat = {'pruning':'trimming', 'selection':'batchnorm', 'reset':'rewind'}
set_criteria_inf = {'pruning':'trimming', 'selection':'information', 'reset':'rewind'}
set_criteria_int = {'pruning':'trimming', 'selection':'info-target', 'reset':'rewind'}
compare_sets([set_criteria_mag, set_criteria_act, set_criteria_bat, set_criteria_inf], 
             ['magnitude', 'activation', 'gradient', 'information'], 'criteria_rewind')
compare_sets([set_criteria_mag], ['magnitude'], 'criteria_rewind_magnitude')
compare_sets([set_criteria_act], ['activation'], 'criteria_rewind_activation')
compare_sets([set_criteria_bat], ['gradient'], 'criteria_rewind_gradient')
compare_sets([set_criteria_inf], ['information'], 'criteria_rewind_information')
#%%
# Compare selection criteria (local)
set_criteria_mag = {'pruning':'trimming', 'scope':'local', 'selection':'magnitude'}
set_criteria_act = {'pruning':'trimming', 'scope':'local', 'selection':'activation'}
set_criteria_gra = {'pruning':'trimming', 'scope':'local', 'selection':'gradient-min'}
set_criteria_bat = {'pruning':'trimming', 'scope':'local', 'selection':'batchnorm'}
set_criteria_inf = {'pruning':'trimming', 'scope':'local', 'selection':'information'}
set_criteria_int = {'pruning':'trimming', 'scope':'local', 'selection':'info-target'}
compare_sets([set_criteria_mag, set_criteria_act, set_criteria_bat, set_criteria_gra, set_criteria_inf], 
             ['magnitude', 'activation', 'batchnorm', 'gradient', 'information'], 'criteria_local')
compare_sets([set_criteria_mag], ['magnitude'], 'criteria_local_magnitude')
compare_sets([set_criteria_act], ['activation'], 'criteria_local_activation')
compare_sets([set_criteria_gra], ['gradient'], 'criteria_local_gradient')
compare_sets([set_criteria_bat], ['batchnorm'], 'criteria_local_batchnorm')
compare_sets([set_criteria_inf], ['information'], 'criteria_local_information')

set_criteria_mag = {'pruning':'trimming', 'scope':'global', 'selection':'magnitude'}
set_criteria_act = {'pruning':'trimming', 'scope':'global', 'selection':'activation'}
set_criteria_gra = {'pruning':'trimming', 'scope':'global', 'selection':'gradient-min'}
set_criteria_bat = {'pruning':'trimming', 'scope':'global', 'selection':'batchnorm'}
set_criteria_inf = {'pruning':'trimming', 'scope':'global', 'selection':'information'}
set_criteria_int = {'pruning':'trimming', 'scope':'global', 'selection':'info-target'}
compare_sets([set_criteria_mag, set_criteria_act, set_criteria_inf], 
             ['magnitude', 'activation', 'information'], 'criteria_global')
compare_sets([set_criteria_mag], ['magnitude'], 'criteria_global_magnitude')
compare_sets([set_criteria_act], ['activation'], 'criteria_global_activation')
compare_sets([set_criteria_inf], ['information'], 'criteria_global_information')

#%%
set_criteria_mag = {'pruning':'trimming', 'scope':'local', 'selection':'magnitude', 'reset':'rewind'}
set_criteria_act = {'pruning':'trimming', 'scope':'local', 'selection':'activation', 'reset':'rewind'}
set_criteria_gra = {'pruning':'trimming', 'scope':'local', 'selection':'gradient-min', 'reset':'rewind'}
set_criteria_bat = {'pruning':'trimming', 'scope':'local', 'selection':'batchnorm', 'reset':'rewind'}
set_criteria_inf = {'pruning':'trimming', 'scope':'local', 'selection':'information', 'reset':'rewind'}
set_criteria_int = {'pruning':'trimming', 'scope':'local', 'selection':'info-target', 'reset':'rewind'}
compare_sets([set_criteria_mag, set_criteria_act, set_criteria_bat, set_criteria_inf], 
             ['magnitude', 'activation', 'batchnorm', 'information'], 'criteria_local_rewind')
compare_sets([set_criteria_mag], ['magnitude'], 'criteria_local_rewind_magnitude')
compare_sets([set_criteria_act], ['activation'], 'criteria_local_rewind_activation')
compare_sets([set_criteria_bat], ['batchnorm'], 'criteria_local_rewind_batchnorm')
compare_sets([set_criteria_inf], ['information'], 'criteria_local_rewind_information')

set_criteria_mag = {'pruning':'trimming', 'scope':'global', 'selection':'magnitude', 'reset':'rewind'}
set_criteria_act = {'pruning':'trimming', 'scope':'global', 'selection':'activation', 'reset':'rewind'}
set_criteria_gra = {'pruning':'trimming', 'scope':'global', 'selection':'gradient-min', 'reset':'rewind'}
set_criteria_bat = {'pruning':'trimming', 'scope':'global', 'selection':'batchnorm', 'reset':'rewind'}
set_criteria_inf = {'pruning':'trimming', 'scope':'global', 'selection':'information', 'reset':'rewind'}
set_criteria_int = {'pruning':'trimming', 'scope':'global', 'selection':'info-target', 'reset':'rewind'}
compare_sets([set_criteria_mag, set_criteria_act, set_criteria_inf], 
             ['magnitude', 'activation', 'information'], 'criteria_global_rewind')
compare_sets([set_criteria_mag], ['magnitude'], 'criteria_global_rewind_magnitude')
compare_sets([set_criteria_act], ['activation'], 'criteria_global_rewind_activation')
compare_sets([set_criteria_inf], ['information'], 'criteria_global_rewind_information')

 #%%                    
"""
######
######

Now produce embedding bounds figure

######
######
"""
# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def check_bounds(prop_name, bound_values, limit, id_x = None, variants='selection'):
    if (id_x is None):
        id_x = np.ones(len(full_results['test_loss'])) * (29) #np.argmin(full_results['test_loss'], axis = 1)
        print('wut')
        print(id_x)
    cur_variants = param_hash[variants]
    # Find property index
    id_p = plot_name.index(variants)
    cmap = plt.cm.get_cmap('jet', len(cur_variants))
    test_error = full_results['test_loss'][:, id_x]
    test_prop = full_results[prop_name][:, id_x]
    #test_error = np.array([full_results['test_loss'][i, ix] for (i, ix) in enumerate(id_x)])
    #test_prop = np.array([full_results[prop_name][i, ix] for (i, ix) in enumerate(id_x)])
    #truth_vals = test_prop < limit
    #test_error = test_error[truth_vals]
    #test_prop = test_prop[truth_vals]
    print(test_error.shape)
    plt.figure()
    for p in range(len(cur_variants)):
        plt.plot([], c=cmap(p))
        markers = ['o', 's', 'v']
        for m in range(len(model)):
            id_s = np.logical_and((kept_metadata[:, id_p] == p), (kept_metadata[:, 1] == m))
            #id_s = np.logical_and(id_s, (kept_metadata[:, 6] == 1))
            plt.scatter(test_error[id_s], test_prop[id_s], c=cmap(p), s=70, marker=markers[m], edgecolors='k')
        vals_idx = np.linspace(0, kept_metadata.shape[0] - 1, kept_metadata.shape[0])[kept_metadata[:, id_p] == p].astype(int)
        print(vals_idx)
        for p2 in vals_idx:
            cur_name = kept_names[p2].split('_')
            cur_name = cur_name[6]
            #plt.text(test_error[p2], test_prop[p2], cur_name)
    for v in bound_values:
        plt.plot([np.min(test_error) * .9, np.max(test_error) * 1.1], [v, v], '--k')
    # Find Pareto optimal points
    full_points = np.stack([test_error, test_prop], axis=1)
    pareto_set = is_pareto_efficient_simple(full_points)
    plt.scatter(test_error[pareto_set], test_prop[pareto_set], c='r', marker='.', s=70)
    plt.yscale('log')
    plt.legend(cur_variants)
    plt.xlim([np.min(test_error) * .9, np.max(test_error) * 1.1])
    plt.savefig('output/figures/bound_'+prop_name+'.pdf')
    plt.close()
    
idx_sel = -1
# Check the FLOPS
values_flops = [160 * 1e3, 320 * 1e3, 41 * 1e6, 53 * 1e6]
limit_flops = 60 * 1e5
check_bounds('flops', values_flops, limit_flops, idx_sel)
# Check the disk size
values_disk = [128 * 1e3, 256 * 1e3, 256 * 1e6, 1 * 1e9]
limit_disk = 1 * 1e9
check_bounds('parameters', values_disk, limit_disk, idx_sel)
# Check the memory write
values_memory = [8 * 1e3, 16 * 1e3, 512 * 1e6, 1 * 1e9]
limit_memory = 60 * 1e5
check_bounds('memory_write', values_memory, limit_memory, idx_sel)

        
#%%
plots = [datasets, model, type_mod, initialize, pruning, selection, reset, scope]
plot_name = ['dataset','model','type_mod','initialize','pruning','selection','reset','scope']
# Variants to exclude from analysis
exclude_variants = ['model', 'type_mod']
# Now analyze everything
for p in range(len(plot_name)):
    print(' - Analyzing ' + plot_name[p])
    cur_variants = plots[p];
    
    # Now find all cross-parameters configurations
    kept_meta_vars = kept_metadata.copy()
    kept_meta_vars[:, p] = 0
    meta_identical = np.zeros((kept_metadata.shape[0], kept_metadata.shape[0]))
    for i in range(kept_metadata.shape[0]):
        for j in range(i + 1, kept_metadata.shape[0]):
            if (np.sum(np.abs(kept_meta_vars[i, :] - kept_meta_vars[j, :])) == 0):
                meta_identical[i, j] = 1
    axis_legend = []
    N = len(cur_variants)
    cmap = plt.cm.get_cmap('jet', len(cur_variants))
    cur_test_curve = full_results['test_loss']
    # Fill all configurations
    mean_curves, min_curves, max_curves, max_stats = [None] * N, [None] * N, [None] * N, []
    mean_curves_var, min_curves_var, max_curves_var, max_stats_var = [None] * N, [None] * N, [None] * N, [None] * N
    axis_legend_var = [None] * N
    best_names, best_small_names = [None] * N, [None] * N
    for v in range(len(cur_variants)):
        print('   . Variant ' + str(cur_variants[v]))
        var_curves = cur_test_curve[kept_metadata[:, p] == v]
        names = kept_names[kept_metadata[:, p] == v]
        print(names)
        if (len(var_curves) == 0):
            var_curves = np.zeros((2, 2))
            names = ['none']
        # Compute curves
        min_curves[v] = np.min(var_curves, axis=0)
        max_curves[v] = np.max(var_curves, axis=0)
        mean_curves[v] = np.mean(var_curves, axis=0)
        # Find best (across all pruning)
        best_idx = np.argmin(np.min(var_curves, axis=1))
        best_names[v] = (names[best_idx], np.min(var_curves, axis=1)[best_idx])
        # Find best at maximal pruning
        best_idx = np.argmin(var_curves[:, -1])
        best_small_names[v] = (names[best_idx], var_curves[best_idx, -1])
        # Compute stats
        max_stats.append(np.min(var_curves, axis=1))
        axis_legend.append(cur_variants[v])
        # Create sub-lists
        mean_curves_var[v], min_curves_var[v], max_curves_var[v], max_stats_var[v] = [None] * len(plot_name), [None] * len(plot_name), [None] * len(plot_name), [None] * len(plot_name)
        axis_legend_var[v] = [None] * len(plot_name)
        for p2 in range(len(plot_name)):
            N2 = len(plots[p2])
            mean_curves_2, min_curves_2, max_curves_2, max_stats_2 = [None] * N2, [None] * N2, [None] * N2, []
            axis_legend_2 = []
            for v2 in range(len(plots[p2])):
                var_curves = cur_test_curve[np.logical_and((kept_metadata[:, p] == v), (kept_metadata[:, p2] == v2))]
                if (len(var_curves) == 0):
                    var_curves = np.zeros((1, 1))
                # Compute curves
                min_curves_2[v2] = np.min(var_curves, axis=0)
                max_curves_2[v2] = np.max(var_curves, axis=0)
                mean_curves_2[v2] = np.mean(var_curves, axis=0)
                # Compute stats
                max_stats_2.append(np.min(var_curves, axis=1))
                axis_legend_2.append(plots[p2][v2])
            mean_curves_var[v][p2] = mean_curves_2
            max_curves_var[v][p2] = max_curves_2
            min_curves_var[v][p2] = min_curves_2
            max_stats_var[v][p2] = max_stats_2
            axis_legend_var[v][p2] = axis_legend_2
    # Print best
    print('Best setting (across pruning)')
    print(best_names)
    print('Best and smallest setting (maximal pruning)')
    print(best_small_names)
    # Boxplot figure
    plt.figure()
    plt.boxplot(max_stats)
    plt.title(plot_name[p])
    plt.xticks(np.linspace(1, len(cur_variants), len(cur_variants)), axis_legend)
    plt.savefig('output/figures/full/results_'+plot_name[p]+'_boxplot.pdf')
    # 
    plt.figure()
    for v in range(len(cur_variants)):
        plt.plot(mean_curves[v], c=cmap(v), linewidth = 2.5, alpha = 0.85)
    for v in range(len(cur_variants)):
        plt.plot(min_curves[v], c=cmap(v), linewidth = 0.75, alpha = 0.6)
        plt.plot(max_curves[v], c=cmap(v), linewidth = 0.75, alpha = 0.6)
        for i in range(1, len(min_curves[v])):
            plt.fill([i-1, i-1, i, i], [min_curves[v][i-1], max_curves[v][i-1], max_curves[v][i], min_curves[v][i]], c=cmap(v), alpha=0.2, edgecolor=None, linewidth=0)
    plt.legend(axis_legend)
    plt.title(plot_name[p])
    plt.savefig('output/figures/full/results_'+plot_name[p]+'_curves.pdf')
    # Now plot all cross - configurations
    for v in range(len(cur_variants)):
        for p2 in range(len(plot_name)):     
            if (plot_name[p] in exclude_variants):
                continue
            plt.figure()
            plt.boxplot(max_stats_var[v][p2])
            plt.title(plot_name[p])
            cur_vars_2 = plots[p2]
            cmap = plt.cm.get_cmap('jet', len(cur_vars_2))
            plt.xticks(np.linspace(1, len(cur_vars_2), len(cur_vars_2)), axis_legend)
            plt.savefig('output/figures/full/results_'+plot_name[p]+'_'+cur_variants[v]+'_'+plot_name[p2]+'_boxplot.pdf')
            plt.figure()
            mean_curves_var[v][p2]
            for v2 in range(len(cur_vars_2)):
                plt.plot(mean_curves_var[v][p2][v2], c=cmap(v2), linewidth = 2.5, alpha = 0.85)
            for v2 in range(len(cur_vars_2)):
                plt.plot(min_curves_var[v][p2][v2], c=cmap(v2), linewidth = 0.75, alpha = 0.6)
                plt.plot(max_curves_var[v][p2][v2], c=cmap(v2), linewidth = 0.75, alpha = 0.6)
                for i in range(1, len(min_curves_var[v][p2][v2])):
                    plt.fill([i-1, i-1, i, i], [min_curves_var[v][p2][v2][i-1], max_curves_var[v][p2][v2][i-1], max_curves_var[v][p2][v2][i], min_curves_var[v][p2][v2][i]], c=cmap(v2), alpha=0.2, edgecolor=None, linewidth=0)
            plt.legend(axis_legend_var[v][p2])
            plt.title(plot_name[p])
            plt.savefig('output/figures/results_'+plot_name[p]+'_'+cur_variants[v]+'_'+plot_name[p2]+'_curves.pdf')
                    