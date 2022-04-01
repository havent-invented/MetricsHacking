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
    default = 11000,
    help='Dataset size for train'
)
parser.add_argument(
    '-datalentest',
    type=int,
    default = 400,
    help='Dataset size for test'
)
parser.add_argument(
    '-met',
    type=str,
    help='provide metrics name'
)

parser.add_argument(
    '-proxy',
    type=str,
    default = "mse",
    help='provide proxy name'
)
parser.add_argument(
    '-k',
    type=int,
    default = 0,
    help='k for mse'
)
parser.add_argument(
    '-codec',
    type=str,
    default = "cheng2020_attn_quality2",
    help='Choose codec'
)
parser.add_argument(
    '-enhance',
    type=str,
    default = "None",
    help='Choose whether to use enhance'
)
parser.add_argument(
    '-optimizeimg',
    type = int,
    default = 0,
    help='Optimize image'
)
parser.add_argument(
    '-quality',
    type = int,
    default = 3,
    help='Codec quality'
)
parser.add_argument(
    '-epochs',
    type=int,
    default = 12,
    help='Number of epochs'
)
parser.add_argument(
    '-patchsz',
    type=int,
    default = 256,
    help='Patch size'
)
parser.add_argument(
    '-batchsz_train',
    type=int,
    default = 4,
    help='Train batch size'
)
parser.add_argument(
    '-batchsz_test',
    type=int,
    default = 4,
    help='Test batch size'
)
args_p = parser.parse_args()

met_name = args_p.met
print(met_name)


exec(open('Current_model_lib.py').read())
if args_p.k != 0:
    target_lst = [met_name, args_p.proxy]
    k_lst = [1, args_p.k]
else:
    target_lst = [met_name]
    k_lst = [1]
if args_p.patchsz != 256:
    patch_sz = args_p.patchsz
elif target_lst[0] == "NIMA":
    patch_sz = 224
elif target_lst[0] == "KONIQ":
    patch_sz = 299
else:
    patch_sz = 256


loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst)
loss_calc = loss_calc.eval().requires_grad_(True).to(device)

max_epoch = args_p.epochs
if args_p.codec == "cheng2020_attn_quality2":
    net_codec = None #codec_Identity
elif args_p.codec == "No":
    net_codec = codec_Identity
else:
    net_codec = None

save_filename = "vimeo11k_" + target_lst[0] + "_" + ("cheng2020_attn_quality2" if net_codec == None else "no_codec") + "_FRtuning"
batch_size_train = args_p.batchsz_train
batch_size_test = args_p.batchsz_test

if args_p.enhance == "None":
    net_enhance = None
elif args_p.enhance == "Identity":
    net_enhance = enhance_Identity
save_net_enhance = True
datalen_test = args_p.datalentest
datalen_train = args_p.datalentrain
optimize_image = args_p.optimizeimg
config_dictionary = { "datalen_test" : datalen_test,
    "datalen_train" : datalen_train,
    "batch_size_train" : batch_size_train,
    "datalen_test" : datalen_test,
    "save_filename" : save_filename,
    "codec" : args_p.codec,
    "k_lst": k_lst,
    "met_name":met_name,
    "max_epoch": max_epoch}

exec(open('Train_current_model.py').read())