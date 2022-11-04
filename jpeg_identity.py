import os
for q in [5, 20, 30, 40, 50, 60, 70, 80]:
    exec_str = f'bsub -gpu "num=1:mode=exclusive_process"  -q normal -W 3:00 -o log.out   python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 0 -met DISTS -k 0 -comment jpeg16_Identity -proxy bpp_loss -name jpeg16_{q}_DISTS_Identity -quality {q} -codec jpeg16 -use_wandb 0   -sigmoid_activation 0  -max_epoch 2   -enhance_net No' 
    os.system(exec_str)

