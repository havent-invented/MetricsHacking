
#pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
#pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_Identity" ,'codec' :  "jpeg",}}
import subprocess
import os
ckpt_recovery = 0 
home_dir_base = "E:/Stuff/MetricsHacking_HDD/tmp_dir/"
new_mets = ["ADISTS", "GMSD", "VTAMIQ", "AHIQ", "IQT", "CONITRIQUE", "MRperceptual", "IQAConformerBNS", "STLPIPS"]
long_mets = ["DISTS", "LPIPS","VIFLoss","HaarPSI", "PieAPP", "ADISTS", "GMSD", "VTAMIQ", "AHIQ", "IQT", "CONITRIQUE", "MRperceptual", "IQAConformerBNS", "STLPIPS"]
met_names = new_mets
q_s = [5,10,20,]#40,60]
#q_s = [5,10,20,40,60]

#launcher = 'bsub -gpu "num=1:mode=exclusive_process" -q normal -W 3:00 -o log_jpeg.out '
launcher = ''
#Identity
if 0:
    for q in q_s:
        for met in met_names: 
            additional_opt_str = " -use_wandb 1 -enhance_net No -max_epoch 2 -sigmoid_activation 0"
            exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 0 -met {met} -k 0 -comment jpeg -proxy bpp_loss -name jpeg_{q}_{met}_Identity -quality {q} -codec jpeg -ckpt_recovery {ckpt_recovery} {additional_opt_str}  -logs_dir E:/Stuff/MetricsHacking_HDD/logs/  '
            os.system(exec_str)
            print(exec_str)
#Hacking train
flag = 0
if 0:
    for q in q_s:
        for met in met_names: 
            if met == "MRperceptual":
                flag = 1
            if flag:
                additional_opt_str = " -use_wandb 1 "
                exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 0 -met {met} -k 0 -comment jpeg -proxy bpp_loss -name jpeg_{q}_{met} -quality {q} -codec jpeg -use_wandb 1 -ckpt_recovery {ckpt_recovery} {additional_opt_str}  -logs_dir E:/Stuff/MetricsHacking_HDD/logs/  -max_epoch 15 '
                os.system(exec_str)
                print(exec_str)
		
#Patches, 10 20; violin plot: [5,10,20,40,60]
if 0:
    import subprocess
    import os
    import random
    ckpt_recovery = 1 
    
    q_s_subjectify = {10,20}
    for met in met_names:
        for q in q_s:
            sav_img = int(q in q_s_subjectify)
            additional_opt_str = " -use_wandb 1 -max_epoch 2 -train_mode 0 -logs_dir E:/Stuff/MetricsHacking_HDD/log_patches/ "
            home_dir = os.path.join(home_dir_base, str(int(random.random()* 1e15)) + "/")
            os.system(f"mkdir {home_dir}")
            exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -mode test -order_pre_post 0 -met {met} -k 0 -comment patches_diffjpeg -proxy SSIM -name patches_jpeg_{q}_{met} -quality {q} -use_wandb 1 -ckpt_recovery {ckpt_recovery} -ckpt_recovery_path E:/Stuff/MetricsHacking_HDD/logs/jpeg_{q}_{met}/ckpt.ckpt {additional_opt_str}  -home_dir {home_dir}  -save_net_enhance 0 -save_ckpt 0 -codec jpeg  -dataset_dir E:/Stuff/MetricsHacking_HDD/datasets/eval_dataset_vimeo_test500/   -save_out {sav_img} -save_enh {sav_img}'
            os.system(exec_str)
            print(exec_str)
    

    ckpt_recovery = 0
    
    for met in met_names: #["PieAPP", "FSIM", "StyleLoss", "NLPD", "ContentLoss", ]:
        for q in q_s:
            additional_opt_str = " -use_wandb 1 -max_epoch 2 -train_mode 0 -logs_dir E:/Stuff/MetricsHacking_HDD/log_patches/ "
            home_dir = os.path.join(home_dir_base, str(int(random.random()* 1e15)) + "/")
            os.system(f"mkdir {home_dir}")
            exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -mode test -order_pre_post 0 -met {met} -k 0 -comment patches_diffjpeg_Identity -proxy SSIM -name patches_jpeg_{q}_{met}_Identity -quality {q} -use_wandb 1 -ckpt_recovery {ckpt_recovery} -ckpt_recovery_path No {additional_opt_str}  -home_dir {home_dir}  -save_net_enhance 0 -save_ckpt 0 -codec jpeg  -dataset_dir E:/Stuff/MetricsHacking_HDD/datasets/eval_dataset_vimeo_test500/   -save_out {sav_img} -save_enh {sav_img}  -enhance_net No  -sigmoid_activation 0 '
            os.system(exec_str) # -codec jpeg_real_pil 
            print(exec_str)
    


#Hacking detection, heatmap
#MRperceptual
test_mets = " DISTS LPIPS VIFLoss HaarPSI PieAPP NLPD mse SSIM ADISTS GMSD AHIQ IQT  " # STLPIPS VTAMIQ IQAConformerBNS CONITRIQUE
ks = " 0 " * 17
q = 5#!
ckpt_recovery = 1
met_names = ["IQT", "CONITRIQUE", "MRperceptual", "IQAConformerBNS", "STLPIPS"]
for met in met_names:#long_mets: 
    additional_opt_str = " -batch_size_train 2 -batch_size_test 4  -use_wandb 1 -max_epoch 2 -train_mode 0 -logs_dir E:/Stuff/MetricsHacking_HDD/logs_hackingdetection/ "    #-num_workers 4 
    exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -mode test -order_pre_post 0 -mets {test_mets} -ks {ks} -comment hackingdetection_diffjpeg -proxy SSIM -name hackingdetection_jpeg_{q}_{met} -quality {q} -use_wandb 1 -ckpt_recovery {ckpt_recovery} -ckpt_recovery_path E:/Stuff/MetricsHacking_HDD/logs/jpeg_{q}_{met}/ckpt.ckpt {additional_opt_str}  -save_net_enhance 0 -save_ckpt 0 -codec jpeg  -dataset_dir E:/Stuff/MetricsHacking_HDD/datasets/eval_dataset_vimeo_test500/   -save_out 0 -save_enh 0'
    os.system(exec_str) # -codec jpeg_real_pil 
    print(exec_str)
ckpt_recovery = 0 
for met in met_names:#long_mets: 
    additional_opt_str = " -batch_size_train 2 -batch_size_test 4  -use_wandb 1 -max_epoch 2 -train_mode 0 -logs_dir E:/Stuff/MetricsHacking_HDD/logs_hackingdetection/ " #-num_workers 4 
    exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -mode test -order_pre_post 0 -mets {test_mets} -ks {ks} -comment hackingdetection_diffjpeg_Identity -proxy SSIM -name hackingdetection_jpeg_{q}_{met}_Identity -quality {q} -use_wandb 1 -ckpt_recovery {ckpt_recovery} -ckpt_recovery_path No {additional_opt_str} -save_net_enhance 0 -save_ckpt 0 -codec jpeg  -dataset_dir E:/Stuff/MetricsHacking_HDD/datasets/eval_dataset_vimeo_test500/   -save_out 0 -save_enh 0  -enhance_net No  -sigmoid_activation 0 '
    os.system(exec_str) # -codec jpeg_real_pil 
    print(exec_str)

#Real JPEG, quality 10
if 0:
    met_names = ["ADISTS", "GMSD", "VTAMIQ", "AHIQ", "IQT", "CONITRIQUE", "MRperceptual", "IQAConformerBNS", "STLPIPS"]
    
    import random
    ckpt_recovery = 0
    q = 10
    
    for met in met_names: 
        home_dir = os.path.join(home_dir_base, str(int(random.random()* 1e15)) + "/")
        additional_opt_str = " -use_wandb 1 -max_epoch 2 -train_mode 0 -sigmoid_activation 0  -max_epoch 2 -save_net_enhance 0 -enhance_net No "
        exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 0 -met {met} -k 0 -comment jpegcv2_Identity -proxy SSIM -name jpegcv2_{q}_{met}_Identity -quality {q} -codec jpeg_real_cv2 -use_wandb 1 -ckpt_recovery {ckpt_recovery} {additional_opt_str} -home_dir {home_dir} -save_ckpt 0  -logs_dir E:/Stuff/MetricsHacking_HDD/logs/  '
        os.system(exec_str) 
        print(exec_str)
        
    import random
    ckpt_recovery = 1 
    
    for met in met_names:
        additional_opt_str = " -use_wandb 1 -max_epoch 2 -train_mode 0"
        home_dir = os.path.join(home_dir_base, str(int(random.random()* 1e15)) + "/")
        os.system(f"mkdir {home_dir}")
        exec_str = f'{launcher} python launch_tuning_fr.py -cfg_dir cfgs/default.yaml -order_pre_post 0 -met {met} -k 0 -comment jpegcv2 -proxy SSIM -name jpegcv2_{q}_{met} -quality {q} -codec jpeg_real_cv2 -use_wandb 1 -ckpt_recovery {ckpt_recovery} -ckpt_recovery_path E:/Stuff/MetricsHacking_HDD/logs/jpeg_{q}_{met}/ckpt.ckpt {additional_opt_str}  -home_dir {home_dir}  -save_net_enhance 0 -save_ckpt 0  -logs_dir E:/Stuff/MetricsHacking_HDD/logs/  '
        os.system(exec_str) 
        print(exec_str) 
    