from __future__ import print_function

import os

import random
import socket
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn

class DummyArg(object):
    """
    Just a dummy arg class. Sometimes it is necessary.
    """
    pass

# The parser object.
parser = ArgumentParser(description='The OD-test evaluation framework.')

parser.add_argument('--no-visualize', default=False, action='store_true', help='Disable visdom visualization. (default False)')

parser.add_argument('--seed', type=int, default=42, help='Random seed. (default 42)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default 128)')
parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers. (default 4)')

parser.add_argument('--exp', '--experiment_id', type=str, default='test', help='The Experiment ID. (default test)')
parser.add_argument('--save', default=False, action='store_true', help='Save the output? (default False)')
parser.add_argument('--no-cuda', default=False, action='store_true', help='Disable cuda?')
parser.add_argument('--cuda-device', type=int, default=0, help='Select cuda device. (default 0)')

parser.add_argument('--force-train-h', default=False, action='store_true', help='Whether should forcibly train H or just reuse.')
parser.add_argument("--force-run", default=False, action='store_true', help='Force run the evaluation experiment?')

args = parser.parse_args()
args.experiment_id = args.exp

# torch.device object is used throughout this script
if not args.no_cuda:
    args.device = torch.device("cuda", index=args.cuda_device)
else:
    args.device = torch.device("cpu")

print("Device:" + str(args.device))

assert torch.cuda.is_available(), 'A cuda device is required!'

# Reproducability.
# Set up the random seed based on the arg.
random.seed(args.seed)
torch.manual_seed(args.seed)
if not args.no_cuda:
    torch.cuda.set_device(args.cuda_device)
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# Set up the default workspace for each experiment.
exp_data = []
workspace_path = os.path.abspath('workspace')
if not os.path.isdir(workspace_path):
    os.makedirs(os.path.abspath('workspace'))

# Make the experiment folder(s).
# In some usecases you may specify multiple comma separated experiments.
exp_list = args.experiment_id.split(',')
exp_paths = []
for exp_id in exp_list:
    experiments_path = os.path.join(workspace_path, 'experiments', exp_id)
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)

    # Make the experiment subfolders.
    for folder_name in exp_data:
        if not os.path.exists(os.path.join(experiments_path, folder_name)):
            os.makedirs(os.path.join(experiments_path, folder_name))
    exp_paths.append(experiments_path)

if len(exp_list) == 1:
    args.experiment_path = exp_paths[0]
else:
    print('Operating in multi experiment mode.')
    args.experiment_path = exp_paths

args.hostname = socket.gethostname()

print(args)

# Save a copy of the args in the experiment folder.
if isinstance(args.experiment_path, str) and args.experiment_id != 'model_ref':
    # This version of PyTorch cant serialize torch.Device.
    import copy
    save_args = copy.copy(args)
    save_args.device = args.device.__str__()
    torch.save(save_args, os.path.join(experiments_path, 'args.pth'))
