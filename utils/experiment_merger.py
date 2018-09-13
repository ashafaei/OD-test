from __future__ import print_function

from argparse import ArgumentParser
from termcolor import colored
import os
import os.path as path
import sys
import torch

parser = ArgumentParser(description='The Merger Tool of the OD-test evaluation.')
parser.add_argument('--in-exp', default=None, type=str, help='The comma separated list of the input experiments.', required=True)
parser.add_argument('--out-exp', default=None, type=str, help='The output experiment folder.', required=True)
parser.add_argument('--force-out', default=False, action='store_true', help='Force save final results.')
args = parser.parse_args()

workspace_path = os.path.abspath('workspace')
assert os.path.isdir(workspace_path), colored('Have you run setup.py?', 'red')

exp_list = args.in_exp.split(',')
exp_paths = []
results = []
method_count = {}
exp_ids = []

def does_match(record_a, record_b):
    if len(record_a) != len(record_b):
        return False
    if len(record_a) == 0:
        return True
    if record_a[0] == record_b[0]:
        return does_match(record_a[1:], record_b[1:])
    else:
        return False

def is_duplicate(base, new_record):
    for entry in base:
        if does_match(entry, new_record):
            return True
    return False

for exp_id in exp_list:
    results_path = path.join(workspace_path, 'experiments', exp_id, 'results.pth')
    if path.isfile(results_path):
        exp_ids.append(exp_id)
        print('Processing %s'%(colored(results_path, 'yellow')))
        sub_results = torch.load(results_path)
        print('\t There are %d records.'%(len(sub_results)))
        for r in sub_results:
            if is_duplicate(results, r):
                print('Duplicate record %s, terminating.'%r)
                sys.exit(1)
            else:
                results.append(r)
                if method_count.has_key(r[0]):
                    method_count[r[0]] += 1
                else:
                    method_count[r[0]] = 1

print('Total length %d'%(len(results)))
output_folder = path.join(workspace_path, 'experiments', args.out_exp)

if not path.isdir(output_folder):
    os.makedirs(output_folder)
else:
    if not args.force_out:
        print('Choose an empty project, %s already exists.'%args.out_exp)
        sys.exit(1)
    
print('Saving results to %s'%output_folder)

total = 0
for method, count in method_count.iteritems():
    print ('%25s\t%-5s'%(method, colored('%d'%count, 'green')))
    total += count
print('%s'%colored('-'*37, 'red'))
print('%25s\t%-5s'%('Total', colored('%d'%total, 'green')))

torch.save(results, path.join(output_folder, 'results.pth'))
torch.save(exp_ids, path.join(output_folder, 'ids.pth'))