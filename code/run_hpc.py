# -*- coding: utf-8 -*-

import os
import argparse
# Argument Parser
parser = argparse.ArgumentParser()
# Device information
parser.add_argument("--datadir",            default="/scratch/esling/datasets/",    type=str,       help="Directory to find datasets")
parser.add_argument("--output",             default="/scratch/esling/output/",     type=str,       help="Output directory")
parser.add_argument("--dataset",            default="mnist",                        type=str,       help="mnist | cifar10 | fashion_mnist | cifar100 | toy")
parser.add_argument("--model",              default="cnn",          type=str,       help="mlp | cnn | ae | vae | wae | vae_flow")
parser.add_argument('--epochs',             default=200,            type=int,       help='')
parser.add_argument("--prune_it",           default=2,              type=int,       help="Pruning iterations count")
parser.add_argument("--rewind_it",          default=100,            type=int,       help="Pruning iterations count")
parser.add_argument("--prune_percent",      default=20,             type=int,       help="Pruning iterations count")
parser.add_argument("--prune",              default="masking",      type=str,       help="masking | trimming | hybrid")
parser.add_argument("--initialize",         default="xavier",       type=str,       help="classic | xavier | kaiming")
parser.add_argument("--device",             default="cuda",         type=str,       help='')
parser.add_argument('--latent_dims',        default=8,              type=int,       help='')
parser.add_argument('--warm_latent',        default=100,            type=int,       help='')
parser.add_argument('--n_hidden',           default=512,            type=int,       help='')
parser.add_argument('--n_layers',           default=3,              type=int,       help='')
parser.add_argument('--channels',           default=64,             type=int,       help='')
parser.add_argument('--kernel',             default=5,              type=int,       help='')
parser.add_argument('--lr',                 default=1e-3,           type=float,     help='')
parser.add_argument('--n_runs',             default=1,              type=int,       help='')
parser.add_argument('--config_type',        default='full',         type=str,       help='Preset configuration')
parser.add_argument('--machine',            default='graham',       type=str,       help='Machine on which we are computing')
parser.add_argument('--time',               default='0-11:59',      type=str,       help='Machine on which we are computing')
args = parser.parse_args()

# Scripts folder
if not os.path.exists('scripts/output/'):
    os.makedirs('./scripts/output/')
    
def write_basic_script(file, args, out_f="%N-%j.out"):
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --gres=gpu:1            # Request GPU generic resources\n")
    if (args.machine == 'cedar'):
        file.write("#SBATCH --cpus-per-task=6   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.\n")
        file.write("#SBATCH --mem=32000M        # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.\n")
    else:
        file.write("#SBATCH --cpus-per-task=16   # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.\n")
        file.write("#SBATCH --mem=64000M         # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.\n")
    file.write("#SBATCH --time=%s\n"%(args.time))
    file.write("#SBATCH --output=" + out_f + "\n")
    file.write("\n")
    file.write("module load python/3.7\n")
    if (args.machine != 'local'):
        file.write("virtualenv --no-download $SLURM_TMPDIR/env\n")
        file.write("source $SLURM_TMPDIR/env/bin/activate\n")
        file.write("pip install --no-index --upgrade pip\n")
        file.write("pip install --no-index -r requirements.txt\n")
        # This is ugly as fuck ... but mandatory
        file.write("pip install ~/scratch/python_libs/nnAudio-0.1.0-py3-none-any.whl\n")
        file.write("pip install ~/scratch/python_libs/SoundFile-0.10.3.post1-py2.py3-none-any.whl\n")
        file.write("pip install ~/scratch/python_libs/resampy-0.2.2.tar.gz\n")
        file.write("pip install ~/scratch/python_libs/librosa-0.7.2.tar.gz\n")
        file.write("pip install ~/scratch/python_libs/lmdb-0.98.tar.gz\n")
        file.write("pip install ~/scratch/python_libs/mir_eval-0.6.tar.gz\n")
        file.write("cd $SLURM_TMPDIR\n")
    else:
        file.write("source $HOME/env/bin/activate\n")
    file.write("\n")
    file.write("cd /scratch/esling/lottery/\n")

# Parse the arguments
args = parser.parse_args()
# Dataset argument
datasets = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
toy_datasets = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
# Models grid arguments
# model = ['mlp', 'cnn', 'ae', 'vae', 'rnn', 'lstm', 'gru']
model = ['sing_ae', 'ddsp', 'wavenet']
# Types of sub-layers in the *AE architectures
type_mod = ['mlp', 'gated_mlp', 'cnn', 'res_cnn', 'gated_cnn']
# Pruning process arguments
initialize = ['classic', 'uniform', 'normal', 'xavier','kaiming']
# Type of pruning operations
prune = ['trimming', 'masking']
# Type of reset / rewind
prune_reset = ['reinit', 'rewind']
# Scope of pruning (local / global)
prune_scope = ['local', 'global']
# Selection criterion
prune_selection = ['magnitude', 'batchnorm', 'gradient_min', 'activation', 'information', 'info_target']
# Percent of pruning per type
prune_percents = {
 'masking':30,
 'trimming':25,
 'hybrid':15
 }

# Using list comprehension to compute all possible permutations 
res = [[i, j, k, l] for i in prune  
                    for j in prune_reset 
                    for k in prune_scope
                    for l in prune_selection]

skipping_conf = {'cl'}

# Set of automatic configurations
class config:
    prune_it        =   [2,     15,             30]
    rewind_it       =   [1,     args.rewind_it, 20]
    n_hidden        =   [64,    args.n_hidden,  1024]
    n_layers        =   [3,     args.n_layers,  6]
    channels        =   [32,    args.channels,  128]
    kernel          =   [5,     args.kernel,    5]
    encoder_dims    =   [16,    16,     64]
    eval_interval   =   [1,     70,     50]

configurations = {'test':0, 'full':1, 'large':2}
final_config = configurations[args.config_type]

def useless_combination(prune, select):
    unused = {'masking':['batchnorm', 'activation', 'information', 'info_target'], 'trimming':'increase', 'hybrid':'increase'}
    return (select in unused[prune])

run_name = 'run_' + args.model + '_' + args.dataset + '.sh'
with open(run_name, 'w') as file:
    cpt = 0
    for r in range(args.n_runs):
        for vals in res:
            if (useless_combination(vals[0], vals[3])):
                continue
            # Write the original script file
            final_script = 'scripts/sc_'  + args.model + '_' +  args.dataset + '_' + str(cpt) + '.sh'
            f_script = open(final_script, 'w')
            write_basic_script(f_script, args, 'output/out_'  + args.model + '_' +  args.dataset + '_' + str(cpt))
            # Write the python launch command
            cmd_str = 'python main.py --device ' + args.device
            cmd_str += ' --datadir ' + args.datadir
            cmd_str += ' --output ' + args.output
            cmd_str += ' --dataset ' + args.dataset
            cmd_str += ' --model ' + args.model
            cmd_str += ' --epochs ' + str(args.epochs)
            cmd_str += ' --prune ' + vals[0]
            cmd_str += ' --prune_reset ' + vals[1]
            cmd_str += ' --prune_scope ' + vals[2]
            cmd_str += ' --prune_selection ' + vals[3]
            cmd_str += ' --prune_percent ' + str(prune_percents[vals[0]])
            cmd_str += ' --warm_latent ' + str(args.warm_latent)
            cmd_str += ' --lr ' + str(args.lr)
            if (args.model == 'wavenet'):
                cmd_str += ' --batch_size 8 '
                cmd_str += ' --rewind_it 7 '
            for n in vars(config):
                if (n[0] != '_'):
                    cmd_str += ' --' + n + ' ' + str(vars(config)[n][final_config])
            cmd_str += ' --k_run ' + str(r)
            f_script.write(cmd_str + '\n')
            f_script.close()
            file.write('sbatch ' + final_script + '\n')
            cpt += 1
os.system('chmod +x ' + run_name)