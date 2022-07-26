import argparse
import random
random.seed(1)
import numpy as np
np.random.seed(1)
import torch
torch.manual_seed(1)
parser = argparse.ArgumentParser(description='Tuning to metrics')
import yaml
from types import SimpleNamespace
with open("cfgs/default.yaml") as fh:
    cfg_tmp = yaml.load(fh, Loader=yaml.FullLoader)

for key, val in cfg_tmp["general"].items():
    if not f"-{key}" in vars(parser)['_option_string_actions']:
        parser.add_argument(
        f'-{key}',
        type=type(val),
        default = None,
        help=key)
        

cfg = None


parser.add_argument(
    '-k',
    type=float,
    default = None,
    help='k for proxy'
)

args_p = parser.parse_args()

with open(args_p.cfg_dir) as fh:
    cfg = yaml.load(fh, Loader=yaml.FullLoader)
    cfg["general"]["cfg_dir"] = args_p.cfg_dir
    #cfg = SimpleNamespace(**cfg)
    for k in cfg.keys():
        if k in args_p.__dict__.keys():
            cfg[k] = args_p.__dict__[k]

if args_p.met != None:
    cfg["general"]["met_names"] =[args_p.met,]

exec(open('Current_model_lib.py').read())
if args_p.k != None:
    cfg["general"]["met_names"] = [args_p.met, args_p.proxy]
    cfg["general"]["k_lst"] = [1, args_p.k]

"""
if args_p.patchsz != 256:
    patch_sz = args_p.patchsz
elif target_lst[0] == "NIMA":
    patch_sz = 224
elif target_lst[0] == "KONIQ":
    patch_sz = 299
else:
    patch_sz = 256
"""

save_net_enhance = True
exec(open('Train_current_model.py').read())
