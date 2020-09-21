import matplotlib.pyplot as plt
import pretty_midi
import random
import numpy as np
import os
import torch
from torch import distributions
import argparse
# from models.encoders import *
# from models.ae import *


def reconstruction(args, model, epoch, dataset):
    # Plot settings
    nrows, ncols = 4, 2  # array of sub-plots
    figsize = np.array([8, 20])  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # generate random index for testing random data
    rand_ind = np.array([random.randint(0, len(dataset) - 1) for i in range(nrows)])
    ind = 0
    for i, axi in enumerate(ax.flat):
        if i % 2 == 0:
            piano_roll = dataset[rand_ind[ind]]
            axi.matshow(piano_roll, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Original number " + str(rand_ind[ind]))
        else:
            cur_input = dataset[rand_ind[ind]].unsqueeze(0).to(args.device)
            x_reconstruct, _, _ = model(cur_input)
            x_reconstruct = x_reconstruct[0].detach().cpu()
            if args.num_classes > 1:
                x_reconstruct = torch.argmax(x_reconstruct, dim=0)
            axi.matshow(x_reconstruct, alpha=1)
            # write row/col indices as axes' title for identification
            axi.set_title("Reconstruction number " + str(rand_ind[ind]))
            ind += 1
    plt.tight_layout(True)
    plt.savefig(args.figures_path + '/prune_' + str(args.prune_it) + '_epoch_' + str(epoch))
    plt.close()


def sampling(args, model, fs=25, program=0):
    # Create normal distribution representing latent space
    latent = distributions.normal.Normal(torch.tensor([0], dtype=torch.float),
                                         torch.tensor([1], dtype=torch.float))
    # Sampling random from latent space
    z = latent.sample(sample_shape=torch.Size([args.nb_samples, args.latent_size])).squeeze(2)
    z = z.to(args.device)
    # Pass through the decoder
    generated_bar = model.decode(z)
    # Generate figure from sampling
    generated_bar = generated_bar.detach().cpu()
    if args.num_classes > 1:
        generated_bar = torch.argmax(generated_bar, dim=1)
    for i in range(args.nb_samples):
        plt.matshow(generated_bar[i], alpha=1)
        plt.title("Sampling from latent space")
        plt.savefig(args.figures_path + '/prune_' + str(args.prune_it) + '_sampling' + str(i) + '.png')
        plt.close()
    generated_bar = generated_bar.transpose(0,1).reshape(generated_bar.shape[1], -1)
    # Generate MIDI from sampling
    pm = pretty_midi.PrettyMIDI()
    notes, frames = generated_bar.shape
    instrument = pretty_midi.Instrument(program=program)
    # Pad 1 column of zeros to acknowledge initial and ending events
    piano_roll = np.pad(generated_bar.detach(), [(0, 0), (1, 1)], 'constant')
    # Use changes in velocities to find note on/note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # Keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)
    plt.figure()
    plt.matshow(piano_roll)
    for time, note in zip(*velocity_changes):
        # Use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = 75
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note + args.min_pitch,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    # Write out the MIDI data
    pm.write(args.midi_results_path + '/prune_' + str(args.prune_it) + "_sampling.mid")


def interpolation(args, model, dataset, fs=25, program=0):
    x_a, x_b = dataset[random.randint(0, len(dataset) - 1)], dataset[random.randint(0, len(dataset) - 1)]
    x_a, x_b = x_a.to(args.device), x_b.to(args.device)
    # Encode samples to the latent space
    z_a, z_b = model.encode(x_a.unsqueeze(0)), model.encode(x_b.unsqueeze(0))
    # Run through alpha values
    interp = []
    alpha_values = np.linspace(0, 1, args.n_steps)
    for alpha in alpha_values:
        z_interp = (1 - alpha) * z_a[0] + alpha * z_b[0]
        interp.append(model.decode(z_interp))
    # Draw interpolation step by step
    i = 0
    stack_interp = []
    for step in interp:
        if args.num_classes > 1:
            step = torch.argmax(step[0], dim=0)
        stack_interp.append(step)
        # plt.matshow(step.cpu().detach(), alpha=1)
        # plt.title("Interpolation " + str(i))
        # plt.savefig(args.figures_path + "interpolation" + str(i) + ".png")
        # plt.close()
        i += 1
    stack_interp = torch.cat(stack_interp, dim=1)
    # Draw stacked interpolation
    plt.figure()
    plt.matshow(stack_interp.cpu(), alpha=1)
    plt.title("Interpolation")
    plt.savefig(args.figures_path + '/prune_' + str(args.prune_it) + "_interpolation.png")
    plt.close()
    # Generate MIDI from interpolation
    pm = pretty_midi.PrettyMIDI()
    notes, frames = stack_interp.shape
    instrument = pretty_midi.Instrument(program=program)
    # Pad 1 column of zeros to acknowledge initial and ending events
    piano_roll = np.pad(stack_interp.cpu().detach(), [(0, 0), (1, 1)], 'constant')
    # Use changes in velocities to find note on/note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)
    # Keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)
    for time, note in zip(*velocity_changes):
        # Use time + 1s because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = 75
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note + args.min_pitch,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    # Write out the MIDI data
    pm.write(args.midi_results_path + '/prune_' + str(args.prune_it) + "_interpolation.mid")


if __name__ == "__main__":
    # %%
    # -----------------------------------------------------------
    #
    # Argument parser, get the arguments, if not on command line, the arguments are default
    #
    # -----------------------------------------------------------
    parser = argparse.ArgumentParser(description='PyraProVAE')
    # Device Information
    parser.add_argument('--device', type=str, default='cuda:0', help='device cuda or cpu')
    # Data Parameters
    parser.add_argument('--midi_path', type=str, default='/fast-1/mathieu/datasets', help='path to midi folder')
    parser.add_argument("--test_size", type=float, default=0.2, help="% of data used in test set")
    parser.add_argument("--valid_size", type=float, default=0.2, help="% of data used in valid set")
    parser.add_argument("--dataset", type=str, default="nottingham",
                        help="maestro | nottingham | bach_chorales | midi_folder")
    parser.add_argument("--shuffle_data_set", type=int, default=1, help='')
    # Novel arguments
    parser.add_argument('--frame_bar', type=int, default=64, help='put a power of 2 here')
    parser.add_argument('--score_type', type=str, default='mono', help='use mono measures or poly ones')
    parser.add_argument('--score_sig', type=str, default='4_4', help='rhythmic signature to use (use "all" to bypass)')
    parser.add_argument('--data_normalize', type=int, default=1, help='normalize the data')
    parser.add_argument('--data_binarize', type=int, default=1, help='binarize the data')
    parser.add_argument('--data_pitch', type=int, default=1, help='constrain pitches in the data')
    parser.add_argument('--data_export', type=int, default=0, help='recompute the dataset (for debug purposes)')
    parser.add_argument('--data_augment', type=int, default=1, help='use data augmentation')
    # Model Saving and reconstruction
    parser.add_argument('--output_path', type=str, default='output', help='major path for data output')
    # Model Parameters
    parser.add_argument("--model", type=str, default="vae", help='ae | vae | vae-flow | wae')
    parser.add_argument("--beta", type=float, default=1., help='value of beta regularization')
    parser.add_argument("--beta_delay", type=int, default=0, help='delay before using beta')
    parser.add_argument("--encoder_type", type=str, default="gru",
                        help='mlp | cnn | res-cnn | gru | cnn-gru | hierarchical')
    # PyraPro and vae_mathieu specific parameters: dimensions of the architecture
    parser.add_argument('--enc_hidden_size', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--latent_size', type=int, default=128, help='do not touch if you do not know')
    parser.add_argument('--cond_hidden_size', type=int, default=1024, help='do not touch if you do not know')
    parser.add_argument('--cond_output_dim', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--dec_hidden_size', type=int, default=512, help='do not touch if you do not know')
    parser.add_argument('--num_layers', type=int, default=2, help='do not touch if you do not know')
    parser.add_argument('--num_subsequences', type=int, default=8, help='do not touch if you do not know')
    parser.add_argument('--num_classes', type=int, default=2, help='number of velocity classes')
    parser.add_argument('--initialize', type=int, default=0, help='use initialization on the model')
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--subsample', type=int, default=0, help='train on subset')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--nbworkers', type=int, default=3, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # Parse the arguments
    parser.add_argument('--n_steps', type=int, default=11, help='number of steps for interpolation')
    parser.add_argument('--nb_samples', type=int, default=10, help='number of sampling from latent space')
    args = parser.parse_args()

    model_variants = [args.dataset, args.score_type, args.data_binarize, args.num_classes, args.data_augment,
                      args.model, args.encoder_type, args.latent_size, args.beta, args.enc_hidden_size]
    args.final_path = args.output_path
    for m in model_variants:
        args.final_path += str(m) + '_'
    args.final_path = args.final_path[:-1] + '/'
    if os.path.exists(args.final_path):
        os.system('rm -rf ' + args.final_path + '/*')
    else:
        os.makedirs(args.final_path)
    # Create all sub-folders
    args.model_path = args.final_path + 'models/'
    args.tensorboard_path = args.final_path + 'tensorboard/'
    args.weights_path = args.final_path + 'weights/'
    args.figures_path = args.final_path + 'figures/'
    args.midi_results_path = args.final_path + 'midi/'
    for p in [args.model_path, args.tensorboard_path, args.weights_path, args.figures_path, args.midi_results_path]:
        os.makedirs(p)
    # Ensure coherence of classes parameters
    if args.data_binarize and args.num_classes > 1:
        args.num_classes = 2
    # train_loader, valid_loader, test_loader, train_set, valid_set, test_set, args = import_dataset(args)

    print("[DEBUG BEGIN]")
    epoch = 200
    model = torch.load(args.output_path + '/out200/_epoch_' + str(epoch) + '.pth', map_location=torch.device('cpu'))
    sampling(args, model)
    # interpolation(args, model, test_set)
    print("[DEBUG END]")
