import argparse
import uuid
import os
import torch
import json
import numpy as np
import torch.backends.cudnn as cudnn
from utils import AverageMeter
from torch.optim import Adam, lr_scheduler, RMSprop
from torch.utils.data import DataLoader
from ActorCriticNetwork import ActorCriticNetwork
from DataGenerator import TSPDataset
from TSPEnvironment import TSPInstanceEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()

    # ----------------------------------- Data ---------------------------------- #
    parser.add_argument('--train_size',
                        default=5120, type=int, help='Training data size')
    parser.add_argument('--test_size',
                        default=256, type=int, help='Test data size')
    parser.add_argument('--test_from_data',
                        default=True, action='store_true', help='Test data size')
    parser.add_argument('--batch_size',
                        default=512, type=int, help='Batch size')
    parser.add_argument('--n_points',
                        type=int, default=20, help='Number of points in the TSP')

    # ---------------------------------- Train ---------------------------------- #
    parser.add_argument('--n_steps',
                        default=200,
                        type=int, help='Number of steps in each episode')
    parser.add_argument('--n',
                        default=8,
                        type=int, help='Number of steps to bootstrap')
    parser.add_argument('--gamma',
                        default=0.99,
                        type=float, help='Discount factor for rewards')
    parser.add_argument('--render',
                        default=False,
                        action='store_true', help='Render')
    parser.add_argument('--render_from_epoch',
                        default=0,
                        type=int, help='Epoch to start rendering')
    parser.add_argument('--update_value',
                        default=False,
                        action='store_true',
                        help='Use the value function for TD updates')
    parser.add_argument('--epochs',
                        default=200, type=int, help='Number of epochs')
    parser.add_argument('--lr',
                        type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd',
                        default=1e-5,
                        type=float, help='Weight decay')
    parser.add_argument('--beta',
                        type=float, default=0.005, help='Entropy loss weight')
    parser.add_argument('--zeta',
                        type=float, default=0.5, help='Value loss weight')
    parser.add_argument('--max_grad_norm',
                        type=float, default=0.3, help='Maximum gradient norm')
    parser.add_argument('--no_norm_return',
                        default=False,
                        action='store_true', help='Disable normalised returns')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 1)')
    parser.add_argument('--rms_prop',
                        default=False,
                        action='store_true', help='Disable normalised returns')
    parser.add_argument('--adam_beta1',
                        type=float, default=0.9, help='ADAM beta 1')
    parser.add_argument('--adam_beta2',
                        type=float, default=0.999, help='ADAM beta 2')
    # ----------------------------------- GPU ----------------------------------- #
    parser.add_argument('--gpu',
                        default=True, action='store_true', help='Enable gpu')
    parser.add_argument('--gpu_n',
                        default=1, type=int, help='Choose GPU')
    # --------------------------------- Network --------------------------------- #
    parser.add_argument('--input_dim',
                        type=int, default=2, help='Input size')
    parser.add_argument('--embedding_dim',
                        type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_dim',
                        type=int, default=128, help='Number of hidden units')
    parser.add_argument('--n_rnn_layers',
                        type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--n_actions',
                        type=int, default=2, help='Number of nodes to output')
    parser.add_argument('--graph_ref',
                        default=False,
                        action='store_true',
                        help='Use message passing as reference')

    # ----------------------------------- Misc ---------------------------------- #
    parser.add_argument("--name", type=str, default="", help="Name of the run")
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_dir', type=str, default='models')

    print(parser)

    # unique id in case of no name given
    uid = uuid.uuid4()
    id = uid.hex

    # create {} to log stuff
    log = {}
    log['hyperparameters'] = {}
    args = parser.parse_args()

    # log hyperparameters
    for arg in vars(args):
        log['hyperparameters'][arg] = getattr(args, arg)

    # give it a clever name :D
    if args.name != '':
        id = args.name
    print("Name:", str(id))

    USE_CUDA = False
    device = torch.device("cpu")

    policy = ActorCriticNetwork(args.input_dim,  # 2
                                args.embedding_dim,  # 128
                                args.hidden_dim,  # 128
                                args.n_points,  # 20
                                args.n_rnn_layers,  # 1
                                args.n_actions,  # 2
                                args.graph_ref)  # False

    optimizer = Adam(policy.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    train_data = TSPDataset(dataset_fname=None,
                            size=args.n_points,
                            num_samples=args.test_size)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=6)


    batch_sample = next(iter(train_loader))
    print(batch_sample)


    #for batch_idx, batch_sample in train_loader:
      #  print(batch_sample)

    #print(f'Dataset: {train_loader.dataset.__str__()}')
    #print(f' Batch: {train_loader.batch_sampler.sampler}')
    #print(f' BatchSize: {train_loader.batch_size}')
    # print(f' Batch: {train_loader.}')

class buffer:

    def __init__(self):
        # action & reward buffer
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.entropies[:]


def select_action(state, hidden, buffer, best_state):

    probs, action, log_probs_action, v, entropy, hidden = 1#policy(state,
                                                            #     best_state,
                                                             #    hidden)
    buffer.log_probs.append(log_probs_action)
    buffer.states.append(state)
    buffer.actions.append(action)
    buffer.values.append(v)
    buffer.entropies.append(entropy)
    return action, v, hidden


main()