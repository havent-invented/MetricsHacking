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
    default = "mse_loss",
    help='provide proxy name'
)
parser.add_argument(
    '-k',
    type=int,
    default = 20,
    help='k for mse'
)
args_p = parser.parse_args()

k = args_p.k
met_name = args_p.met
print(str(k)+ met_name + " " + args_p.proxy)


patch_sz = 256
exec(open('Current_model_lib.py').read())
target_lst = [met_name, args_p.proxy]
k_lst = [1, k]
loss_calc = Custom_enh_Loss(target_lst = target_lst, k_lst=k_lst)
max_epoch = 12
optimize_image = False
net_codec = codec_Identity
if target_lst[1] == "mse_loss":
    save_filename = "vimeo11k_"+ target_lst[0] +"_" + str(int(k_lst[-1])) + "mse_enhance_"+ ("cheng2020_attn_quality2" if net_codec == None else "no_codec") +"_fixed_direction1103"
else:
    save_filename = "vimeo11k_"+ target_lst[0] +"_" + str(int(k_lst[-1])) + str(target_lst[-1])  + "_enhance_"+ ("cheng2020_attn_quality2" if net_codec == None else "no_codec") +"_fixed_direction1103"
batch_size_train = 4#3
batch_size_test = 4
#net_enhance = enhance_Identity
save_net_enhance = True
datalen_test = 400
datalen_train = 11000

exec(open('Train_current_model.py').read())