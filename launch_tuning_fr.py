import argparse
import yaml
import random
from types import SimpleNamespace
random.seed(1)
import numpy as np
import os
np.random.seed(1)
import torch
torch.manual_seed(1)
parser = argparse.ArgumentParser(description='VQM hacking')
with open("cfgs/default.yaml") as fh:
    cfg_tmp = yaml.load(fh, Loader=yaml.FullLoader)
parser.add_argument(
    '-met',
    type=str,
    default = None,
    help='provide metrics name'
)
parser.add_argument(
    '-proxy',
    type=str,
    default = None,
    help='provide proxy name'
)

parser.add_argument(
    '-k',
    type=float,
    default = None,
    help='k for proxy'
)

for key, val in cfg_tmp["general"].items():
    if not f"-{key}" in vars(parser)['_option_string_actions']:
        parser.add_argument(
        f'-{key}',
        type=type(val),
        default = None,
        help=key)

args_p = parser.parse_args()
with open(args_p.cfg_dir) as fh:
    cfg = yaml.load(fh, Loader=yaml.FullLoader)
cfg["general"]["cfg_dir"] = args_p.cfg_dir

for k in cfg["general"].keys():
    if k in args_p.__dict__.keys() and args_p.__dict__[k] != None:
        cfg["general"][k] = args_p.__dict__[k]


if args_p.met != None:
    cfg["general"]["met_names"] =[args_p.met,]

if args_p.k != None:
    cfg["general"]["met_names"] = [args_p.met, args_p.proxy]
    cfg["general"]["k_lst"] = [1, args_p.k]
print(cfg)
exec(open(os.path.join(cfg["general"]["project_dir"], "Current_model_lib.py")).read())
exec(open('Train_current_model.py').read())