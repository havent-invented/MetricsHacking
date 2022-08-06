import subprocess
import os

#met_names = ["SSIM", "HaarPSI", "MDSI", "VSI", "SRSIM", "GMSD", "VIFLoss", "PieAPP", "FSIM","StyleLoss","NLPD","ContentLoss","DISTS", "LPIPS", ]
#met_names = ["VIFLoss","HaarPSI", "PieAPP", "NLPD", "ContentLoss", "StyleLoss", "DSS", "SRSIM", "SSIM"]# Prior
met_names = ["DISTS", "LPIPS","VIFLoss","HaarPSI", "PieAPP", "NLPD", "ContentLoss", "StyleLoss", "DSS", "SRSIM", "SSIM"]
#for met in ["SRSIM",  "VSI",  "MDSI", "GMSD", "VIF", "VIFs", "VIFLoss", "HaarPSI", "SSIM", "DSS",]:
ckpt_recovery = 0
for met in met_names: #["PieAPP", "FSIM", "StyleLoss", "NLPD", "ContentLoss", ]:
    additional_opt_str = (" -batch_size_train 2 -batch_size_test 4 " if met == "PieAPP" else "")
    additional_opt_str += " -use_wandb 0 " 
    exec_str = f'bsub -gpu "num=1:mode=exclusive_process"  -q normal -W 3:00 -o log.out python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 1 -met {met} -k 0 -comment default_TheRun -proxy bpp_loss -name cheng2020_attn__{met}__default__0__bpp__default__0__mse__1__default_TheRun0508post -quality 1 -codec_metric mse -codec cheng2020_attn -ckpt_recovery {ckpt_recovery} {additional_opt_str}'
    os.system(exec_str)
    print(0)####SRSIM VSI
    exec_str = f'bsub -gpu "num=1:mode=exclusive_process"  -q normal -W 3:00 -o log.out python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 1 -met {met} -k 0 -comment default_TheRun2107 -proxy bpp_loss -name Blur__{met}__default__0__bpp__default__1__2.0__2.0__5__5__default_TheRun0508post -blur_sigma_min 2.0 -blur_sigma_max 2.0 -blur_sz_min 5 -blur_sz_max 5 -codec Blur -ckpt_recovery {ckpt_recovery} {additional_opt_str}'
    os.system(exec_str)    
    print(1)
    exec_str = f'bsub -gpu "num=1:mode=exclusive_process"  -q normal -W 3:00 -o log.out python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 1 -met {met} -k 0.1 -comment default_TheRun -proxy bpp_loss -name cheng2020_attn__{met}__default__0.1__bpp__default__0__mse__1__default_TheRun0508post   -quality 1 -codec_metric mse  -codec cheng2020_attn -ckpt_recovery {ckpt_recovery} {additional_opt_str}'
    os.system(exec_str)
    print(2)
    
