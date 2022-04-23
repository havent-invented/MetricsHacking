import argparse

import random
random.seed(1)
import numpy as np
np.random.seed(1)
import torch
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Tuning to metrics')
parser.add_argument(
    '-datalentrain',
    type=int,
    default = None,
    help='Dataset size for train'
)
parser.add_argument(
    '-datalentest',
    type=int,
    default = None,
    help='Dataset size for test'
)
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
    '-name',
    type=str,
    default = None,
    help='Name'
)
parser.add_argument(
    '-k',
    type=int,
    default = None,
    help='k for proxy'
)
parser.add_argument(
    '-codec',
    type=str,
    default = None,
    help='Choose codec'
)
parser.add_argument(
    '-enhance',
    type=str,
    default = None,
    help='Choose whether to use enhance'
)
parser.add_argument(
    '-optimizeimg',
    type = int,
    default = None,
    help='Optimize image'
)
parser.add_argument(
    '-quality',
    type = int,
    default = None,
    help='Codec quality'
)
parser.add_argument(
    '-epochs',
    type=int,
    default = None,
    help='Number of epochs'
)
parser.add_argument(
    '-patchsz',
    type=int,
    default = None,
    help='Patch size'
)
parser.add_argument(
    '-batchsz_train',
    type=int,
    default = None,
    help='Train batch size'
)
parser.add_argument(
    '-batchsz_test',
    type=int,
    default = None,
    help='Test batch size'
)
parser.add_argument(
    '-cfg_dir',
    type=str,
    default = "",
    help='Config file'
)

args_p = parser.parse_args()
import yaml
from types import SimpleNamespace

if 1 or args_p.cfg_dir != "cfg.yaml":
    with open(args_p.cfg_dir) as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)
        cfg["general"]["cfg_dir"] = args_p.cfg_dir
        #cfg = SimpleNamespace(**cfg)
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

if args_p.epochs != None:
    cfg["general"]["max_epoch"] = args_p.epochs
if args_p.name != None:
    cfg["general"]["name"] = args_p.name
if args_p.codec != None:
    cfg["general"]['codec'] = args_p.codec #codec_Identity
    
    
if args_p.batchsz_train != None:
    cfg["general"]['batch_size_train']= args_p.batchsz_train
if args_p.batchsz_test != None:
    cfg["general"]['batch_size_test'] = args_p.batchsz_test
    
if args_p.enhance != None:
    cfg["general"]['enhance_net'] = args_p.enhance
    if args_p.enhance == "Identity":
        cfg["run"]['net_enhance'] = enhance_Identity

save_net_enhance = True
if args_p.datalentest != None:
    cfg["general"]["datalen_test"] = args_p.datalentest
if args_p.datalentrain != None:
    cfg["general"]["datalen_train"] = args_p.datalentrain
if  args_p.optimizeimg != None:
     cfg["general"]["optimize_image"] = args_p.optimizeimg

if args_p.patchsz != None:
    cfg["general"]['patch_sz'] = args_p.patchsz
exec(open('Train_current_model.py').read())
#%run Train_current_model.py