import argparse

parser = argparse.ArgumentParser(description='Tuning to metrics')
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
    default = 20,
    help='k for mse'
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

k = args_p.k
met_name = args_p.met
print(str(k)+ met_name + " " + args_p.proxy)



exec(open('Current_model_lib.py').read())
target_lst = [met_name, args_p.proxy]
k_lst = [1, k]
loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst)
max_epoch = 12
optimize_image = False
net_codec = codec_Identity
if target_lst[1] == "mse":
    save_filename = "vimeo11k_"+ target_lst[0] +"_" + str(int(k_lst[-1])) + "mse_enhance_"+ ("cheng2020_attn_quality2" if net_codec == None else "no_codec") +"_fixed_direction1103"
else:
    save_filename = "vimeo11k_"+ target_lst[0] +"_" + str(int(k_lst[-1])) + str(target_lst[-1])  + "_enhance_"+ ("cheng2020_attn_quality2" if net_codec == None else "no_codec") +"_fixed_direction1103"

batch_size_train = args_p.batchsz_train
batch_size_test = args_p.batchsz_test

if args_p.patchsz != 256:
    patch_sz = args_p.patchsz
elif target_lst[0] == "NIMA":
    patch_sz = 224
elif target_lst[0] == "KONIQ":
    patch_sz = 299
else:
    patch_sz = 256

save_net_enhance = True
datalen_test = 400
datalen_train = 11000

exec(open('Train_current_model.py').read())