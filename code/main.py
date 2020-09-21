# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('Agg')
# Basic libraries
import os
import time

import argparse
import numpy as np
# Torch libraries
import torch
import torch.nn as nn
#from tensorboardX import SummaryWriter
from model import GatedMLP, GatedCNN, SimpleRNN, SimpleCNN1D
from model import LotteryAE, LotteryVAE, LotteryWAE, LotteryVAEFlow
from model import construct_encoder_decoder
from model_classifier import LotteryClassifierCNN
from model_onset import LotteryTranscriptionCNN
# Internal imports
from data import import_dataset
from initialize import InitializationStrategy
from pruning import PruningMasking, PruningTrimming, PruningHybrid
from analysis import Analyzer

#%%
# Argument Parser
parser = argparse.ArgumentParser()
# Data parameters
parser.add_argument("--datadir",            default="/Users/esling/Datasets/symbolic/",type=str,   help="Directory to find datasets")
parser.add_argument("--dataset",            default="nottingham",       type=str,       help="mnist | cifar10 | fashion_mnist | cifar100 | toy")
parser.add_argument("--test_size",      type=float, default=0.2,        help="% of data used in test set")
parser.add_argument("--shuffle_data_set", type=int, default=1,          help='')
# Model parameters
parser.add_argument("--model",              default="vae",   type=str,       help="mlp | gated_mlp | cnn | gated_cnn | res_cnn")
parser.add_argument('--n_hidden',           default=110,            type=int,       help='Number of FC hidden units')
parser.add_argument('--n_layers',           default=4,              type=int,       help='Number of layers')
# CNN parameters
parser.add_argument('--channels',           default=64,             type=int,       help='')
parser.add_argument('--kernel',             default=8,              type=int,       help='')
parser.add_argument('--dilation',           default=1,              type=int,       help='')
# AE-specific parameters
parser.add_argument('--type_mod',           default='cnn-gru',          type=str,       help='')
parser.add_argument('--encoder_dims',       default=16,             type=int,       help='')
parser.add_argument('--latent_dims',        default=3,              type=int,       help='')
parser.add_argument('--warm_latent',        default=100,            type=int,       help='')
parser.add_argument('--beta_factor',        default=1,              type=int,       help='')
# SING-AE-specific parameters
parser.add_argument("--pad",                default=2304,           type=int,       help="Extra padding added to the waveforms")
parser.add_argument("--sample_rate",        default=16000,          type=int,       help='')
# Lottery ticket parameters
parser.add_argument("--initialize",         default="xavier",       type=str,       help="normal | xavier")
parser.add_argument("--prune",              default="trimming",     type=str,       help="masking | trimming")
parser.add_argument("--prune_selection",    default="information",  type=str,       help="magnitude | batchnorm")
parser.add_argument("--prune_reset",        default="rewind",       type=str,       help="lt | reinit")
parser.add_argument("--prune_scope",        default="global",       type=str,       help="local | global")
parser.add_argument("--prune_scale",        default="normalize",    type=str,       help="local | global")
parser.add_argument("--prune_percent",      default=20,             type=int,       help="Pruning percent")
parser.add_argument("--prune_it",           default=15,             type=int,       help="Pruning iterations count")
parser.add_argument("--rewind_it",          default=1,              type=int,       help="Rewinding iteration")
parser.add_argument("--light_stats",        default=1,              type=int,       help="Light stats allow to use less memory")
# Optimization parameters
parser.add_argument('--train_type',         default='fixed',        type=str,       help='Type of training (fixed for removing randomness)')
parser.add_argument("--lr",                 default=1e-3,           type=float,     help="Initial learning rate")
parser.add_argument("--momentum",           default=.9,             type=float,     help="Initial learning rate")
parser.add_argument("--batch_size",         default=64,             type=int,       help='')
parser.add_argument('--k_run',              default=0,              type=int,       help='')
parser.add_argument('--early_stop',         default=30,             type=int,       help='')
parser.add_argument('--eval_interval',      default=25,             type=int,       help='')
parser.add_argument('--epochs',             default=200,            type=int,       help='')
parser.add_argument('--nbworkers',          default=0,              type=int,       help='')
parser.add_argument('--subsample',      type=int, default=0,            help='train on subset')
parser.add_argument('--seed',           type=int, default=1,            help='random seed')
# Printing and output parameters
parser.add_argument("--print_freq",         default=1,              type=int,       help='')
parser.add_argument("--valid_freq",         default=25,             type=int,       help='')
parser.add_argument("--output",             default="output",       type=str,       help='')
# Toy examples parameters
parser.add_argument("--toy_points",         default=1000,           type=int,       help='Number of points generated for toy set')
parser.add_argument("--toy_noise",          default=2,              type=int,       help='Level of noise in toy set generation')
parser.add_argument("--valid_size",         default=0.2,            type=float,     help="Percentage of data used in validation set")
# Device information
parser.add_argument("--device",             default="cpu",          type=str,       help='')
# Parse the arguments
args = parser.parse_args()
args.eps = 1e-9


#%%
"""
###################
Basic definitions
###################
"""
# Track start time (for HPC)
start_time = time.time()
# Enable CuDNN optimization
if args.device != 'cpu':
    torch.backends.cudnn.benchmark=True
# Check combinations of pruning parameters
unused = {'masking':['batchnorm', 'activation', 'information', 'info_target'], 'trimming':'increase', 'hybrid':'increase'}
if args.prune_selection in unused[args.prune]:
    print('*******')
    print('The following pruning combination is not implemented.')
    print('Pruning \t : ' + args.prune)
    print('Selection \t : ' + args.prune_selection)
    print('Please change combination of pruning parameters.')
    print('*******')
    exit(0)
# Set high rewind it (40%)
if args.rewind_it == -1:
    args.rewind_it = int(args.epochs * .4)
# Model save file
name_list = [args.dataset, args.model, args.type_mod, args.initialize, 
             args.prune, args.prune_selection, args.prune_reset, args.prune_scope, 
             args.k_run]
model_name = ''
for n in name_list:
    model_name += str(n) + '_'
model_name = model_name[:-1]
model_save = args.output + '/' + model_name
# Results and checkpoint folders
args.model_save = model_save
args.figures_path = model_save
args.midi_results_path = model_save
if not os.path.exists('{0}'.format(args.model_save)):
    os.makedirs('{0}'.format(args.model_save))
# Handling cuda
args.cuda = not args.device == 'cpu' and torch.cuda.is_available()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('*******')
print('Run info.')
print('Optimization will be on ' + str(args.device) + '.')
print('Model is ' + str(model_name) + '.')
print('*******')


#%%
"""
###################
Data importing
###################
"""
train_loader, valid_loader, test_loader, args = import_dataset(args)
args.min_pitch = args.testset.min_p

#%%
"""
###################
Initialization and pruning strategy
###################
"""
initializer = InitializationStrategy(args)
args.initializer = initializer
# Pruning strategy
if args.prune == 'masking':
    pruning = PruningMasking(args)
elif args.prune == 'trimming':
    pruning = PruningTrimming(args)
elif args.prune == 'hybrid':
    pruning = PruningHybrid(args)
else:
    print("Unknown pruning " + args.pruning + '.\n')
    exit()
args.pruning = pruning

#%%
"""
###################
Model creation
###################
"""
if args.model in ['mlp', 'gated_mlp']:
    args.type_mod = args.model
    model = GatedMLP(args)
elif args.model in ['cnn', 'gated_cnn', 'res_cnn']:
    args.type_mod = args.model
    model = GatedCNN(args)
elif args.model == 'cnn1d':
    args.type_mod = args.model
    model = SimpleCNN1D(args)
elif args.model in ['rnn', 'gru', 'lstm']:
    args.type_mod = args.model
    model = SimpleRNN(args)
elif args.model in ['ae', 'vae', 'vae_flow', 'wae']:
    args.output_size = args.input_size
    args = construct_encoder_decoder(args)
    if args.model == 'ae':
        model = LotteryAE(args)
    elif args.model == 'vae':
        model = LotteryVAE(args)
    elif args.model == 'wae':
        model = LotteryWAE(args)
    elif args.model == 'vae_flow':
        model = LotteryVAEFlow(args)
elif args.model == 'classify':
    model = LotteryClassifierCNN(args)
elif args.model == 'transcribe':
    model = LotteryTranscriptionCNN(args)
else:
    print("Unknown model " + args.model + ".\n")
    exit()
# Send model weights to the device
model.to(args.device)
print(model)

#%%
"""
###################
Initialize model and analyzer save
###################
"""
# Apply weight initialization
model.apply(initializer)
# Create an analyzer object
analyzer = Analyzer(args)

#%%
"""
###################
Create optimizer
###################
"""
# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-6)
# Use cross-entropy loss
if args.model in ['ae', 'vae', 'wae', 'vae_flow']:
    criterion = nn.L1Loss()
elif args.model in ['classify']:
    criterion = nn.CrossEntropyLoss(reduction='mean')
elif args.model in ['transcribe']:
    criterion = nn.CrossEntropyLoss(size_average=False, reduction='none')
else:
    criterion = nn.CrossEntropyLoss()
if args.num_classes > 1:
    criterion = nn.NLLLoss(reduction='sum')

#%%
"""
###################
Optimization procedure
###################
"""
# Create an initial mask
pruning.initialize(model)
remaining_weights = 100
# Lottery iterations
for prune_it in range(args.prune_it):
    if prune_it > 0:
        # Perform one full cumulative epoch 
        if args.prune_selection in ['gradient_min', 'gradient_max', 'activation', 'information', 'info_target']:
            limit = 1e5
            # For heavy criterion limit the number of examples
            if args.prune_selection in ['activation', 'information', 'info_target']:
                limit = (args.model in ['sing_ae', 'wavenet']) and 256 or 1e3
                # Add hooks to the model
                pruning.register_activation_hooks(model)
            print('[Performing one full cumulative epoch]')
            model.cumulative_epoch(valid_loader, optimizer, criterion, limit, args)
            if args.prune_selection in ['activation', 'information', 'info_target']:
                pruning.remove_activation_hooks()
        # Update mask from subnetwork
        model = pruning.step(model)
        # Reset network to original accuracy
        model = pruning.reset(model)
        # Recreate the optimizer
        print(args.lr)
        del optimizer
        del scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # optimizer.reset()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = args.lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-6)
        # Register the effect of the pruning
        if args.light_stats == 0:
            analyzer.register_pruning(model, pruning, prune_it)
    # Call the meta-analyzer prior to training
    analyzer.iteration_start(model, prune_it)
    print('[Current model size]')
    analyzer.print_summary()
    print('[Supermasks testing]')
    # Testing the accuracy of the model prior to training
    loss = model.test_epoch(test_loader, criterion, 0, args)
    print('[Untrained loss : %.4f]'%(loss))
    print('[Starting training]')
    early_stop = 0
    best_loss = np.inf
    best_valid_loss = np.inf
    losses = torch.zeros(args.epochs, 3)  # + (args.epochs * (prune_it == 0)), 3)
    args.prune_it = prune_it
    # Training iterations
    for i in range(args.epochs):  # + (args.epochs * (prune_it == 0))): #- ((args.rewind_it) * (prune_it > 0))):
        # Update beta factor
        args.beta = args.beta_factor * (float(i) / float(max(args.warm_latent, i)))
        # Training dataset epoch
        losses[i, 0] = model.train_epoch(train_loader, optimizer, criterion, i, args)
        # Validation loss
        losses[i, 1] = model.test_epoch(valid_loader, criterion, i, args)
        # Testing loss
        losses[i, 2] = model.test_epoch(test_loader, criterion, i, args)
        # Compare input data and reconstruction
        #if i % 25 == 0:
        #    reconstruction(args, model, i, args.testset)
        if i % 1 == 0:
            print('Epoch %d \t %f \t %f \t %f'%(i, losses[i, 0], losses[i, 1], losses[i, 2]))
        # Take scheduler step
        scheduler.step(losses[i, 1])
        # Model saving
        if losses[i, 1] < best_valid_loss:
            # Save model
            best_valid_loss = losses[i, 1]
            best_loss = losses[i, 2]
            torch.save(model, args.model_save + '/it_' + str(prune_it) + '_best_valid.th')
            early_stop = 0
        # Check for early stopping
        elif args.early_stop > 0:
            early_stop += 1
            if early_stop > args.early_stop:
                print('[Model stopped early]')
                break
        # Periodic evaluation (or debug model)
        if ((i + 1) % args.eval_interval == 0) or (args.epochs == 1):
            args.plot = 'train'
            with torch.no_grad():
                model.evaluate_epoch(test_loader, criterion, i, args)
        #print(i)
        #print(losses[i])
    # Reload the best performing model
    model_load = torch.load(args.model_save + '/it_' + str(prune_it) + '_best_valid.th')
    model.load_state_dict(model_load.state_dict())
    del model_load
    model_load = None
    # Frequency for Printing Accuracy and Loss
    print('Train loss       : %.6f'%losses[i, 0])
    print('Best valid loss  : %.6f'%best_valid_loss)
    print('Best test loss   : %.6f'%best_loss)
    print('Pruning          : %.2f'%((float(100 - args.prune_percent) / 100) ** prune_it))
    # Call the meta-analyzer after training
    analyzer.iteration_end(model, prune_it, i, losses)
    # Eventually remove the model
    if args.light_stats:
        os.system('rm ' + args.model_save + '/it_' + str(prune_it) + '_best_valid.th')
    # Save a temporary part of the model
    torch.save(analyzer, args.model_save + '.th')
# Remove heavy structure (model and pruning)
analyzer.args.pruning = None
# Finally save the analyzer to record all experiments
torch.save(analyzer, args.model_save + '.th')
