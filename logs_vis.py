from cProfile import label
from cmath import log
import numpy as np
import os
import pickle
import yaml
import matplotlib.cm as cm
import argparse
import matplotlib.pyplot as plt
import shutil 

def load_cfg_from_yaml(path):
    with open(path) as fh:
        cfg = yaml.load(fh, Loader=yaml.FullLoader)
    return cfg

def load_log_from_pkl(path):
    with open(path, 'rb') as f:
        log = pickle.load(f)
    return log
    
def filter_by_cfg(cfg, pattern):
    #if(cfg['general']['cfg_dir'] != "cfgs/default.yaml" or not "blur_sigma_max" in cfg['general']):
    #    return 0
    for key1 in pattern.keys():
        if not (key1 in cfg):
            return 0
        for key2 in pattern[key1].keys():
            if not (key2 in cfg[key1]):
                return 0
            if pattern[key1][key2] != cfg[key1][key2]:
                return 0
    #print(0)
    return 1


def filterIdentity_by_cfg(cfg):
    #if(cfg['general']['cfg_dir'] != "cfgs/default.yaml" or not "blur_sigma_max" in cfg['general']):
    #    return 0
    if (cfg['general']['comment'] == "default_TheRun2107" or cfg['general']['comment'] == "default_TheRun") and cfg['general']['enhance_net'] == "No":
        return 1

def get_names(log_dir, f_func, pattern, load_stats = False):
    logs = {"cfgs": {}, "logs" : {}, "stats" : {}}
    idx = 0
    for i in os.listdir(log_dir):
        cfg = load_cfg_from_yaml(os.path.join(log_dir,i, "cfg.yaml"))
        if f_func(cfg, pattern):
            try:
                log = load_log_from_pkl(os.path.join(log_dir, i,"plots.pkl"))
            except Exception:
                continue 
            logs['logs'][i] = log
            logs['cfgs'][i] = cfg
            if load_stats:
                try:
                    logs['stats'][i] = load_log_from_pkl(os.path.join(log_dir, i,"stats.pkl"))
                except Exception:
                    print("0")
                    pass
        idx += 1
    return logs


#pattern3 =  {'general': {'enhance_net' :  "No", }}
#A = get_names("./logs/", filter_by_cfg, pattern3)
#met_nam = A['cfgs'][list(A['cfgs'].keys())[0]]['general']['met_names'][0]
#print(A['logs'][list(A['cfgs'].keys())[0]][met_nam + "_test"][-1])
def log_concat(cfgs):
    cfg_res = {"cfgs": {}, "logs": {}, "stats": {}}
    idx = 0
    for cfg1 in cfgs:
        for key, cfg2_cfgs, cfg2_logs in zip(cfg1['cfgs'].keys(), cfg1['cfgs'].values(), cfg1['logs'].values()):
            cfg_res['cfgs'][key] = cfg2_cfgs
            cfg_res['logs'][key] = cfg2_logs
            if "stats" in cfg1 and key in cfg1['stats']:
                cfg_res['stats'][key] = cfg1['stats'][key]
            idx += 1
    return cfg_res

def plot_all():
    #pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun2107"}}
    #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun"}}
    pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", "datalen_train" : 11000, "datalen_test" : 1500}}
    pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "Blur", "comment" : "default_TheRun"}}
    Log = get_names("./logs/", filter_by_cfg, pattern2)
    Log1 = get_names("./logs/", filter_by_cfg, pattern1)
    Log = log_concat([Log, Log1])
    for log, cfg, key in zip(Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys()):
        pattern = {'general' : {'comment' : "Identity_run",  'met_names' :  [cfg['general']["met_names"][0]], "quality" : cfg['general']["quality"]}}#"codec_metric" : cfg['general']["codec_metric"], "quality" : cfg['general']["quality"]
        #"met_names" : cfg['general']["met_names"],
        met_nam = cfg['general']['met_names'][0]
        val_impr = log[met_nam + "_test"][-1]
        val_impr_bpp = log["bpp_loss_test"][-1]
        Log_I = get_names("./logs/", filter_by_cfg, pattern)
        if len(Log_I['cfgs'].values()) == 0:
            print(f"Error: no Identity for {key}")
        else:
            val_no = Log_I['logs'][list(Log_I['cfgs'].keys())[0]][met_nam + "_test"][-1]
            val_no_bpp = Log_I['logs'][list(Log_I['cfgs'].keys())[0]]["bpp_loss_test"][-1]
            if cfg['general']['comment'] != "Identity_run":
                print(met_nam,"metrics gain:", val_no - val_impr, "metrics base_value:",val_no, "metrics_preprocessed_value:", val_impr, "bpp_gain:",val_no_bpp - val_impr_bpp,"bpp_base_value:", val_no_bpp, "bpp_preprocessed_value:",val_impr_bpp, list(Log_I['cfgs'].keys())[0], key)


def plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "datalen_train" : 11000, "datalen_test" : 1500}}, \
                pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "Blur1", "comment" : "default_TheRun"}}, \
                pattern = {'general' : {'comment' : "Identity_run", "datalen_train" : 11000, "datalen_test" : 1500 }}, \
                fig_nam = "fig.png", test_flag = True, log_dir = "./logs/", all_mets = False, param_cfg_key = None, stat = "last"):
    #pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun2107"}}
    #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun"}}
    #pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "datalen_train" : 11000, "datalen_test" : 1500}}
    #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "Blur1", "comment" : "default_TheRun"}}
    #pattern = {'general' : {'comment' : "Identity_run", "datalen_train" : 11000, "datalen_test" : 1500 }}#"codec_metric" : cfg['general']["codec_metric"], "quality" : cfg['general']["quality"]
    # 'met_names' :  [cfg['general']["met_names"][0]]
    Log = get_names(log_dir, filter_by_cfg, pattern2)
    Log1 = get_names(log_dir, filter_by_cfg, pattern1)
    Log_I =  get_names(log_dir, filter_by_cfg, pattern)
    Log = log_concat([Log, Log1, Log_I])
    RD_curve = {}
    #
    for log, cfg, key in zip(Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys()):
        #print(met_nam)
        if not "general" in cfg.keys() or not "met_names" in cfg['general'].keys():
            continue
        nam = cfg['general']["comment"] + ("_Identity" if not "Identity" in cfg['general']['comment'] and cfg['general']['enhance_net'] == "No" else "")
        

        if all_mets:
            for met in log.keys():
                if stat == "last":
                    val_impr = log[met + ("_test" if test_flag else "")][-1]
                elif stat == "min":
                    print("____")
                    print(log[met + ("_test" if test_flag else "")])
                    val_impr = np.min(log[met + ("_test" if test_flag else "")])
                if not (met in RD_curve):
                    RD_curve[met] = {}
                if not (nam in RD_curve[met]):
                    RD_curve[met][nam] = {}
                if not 'met' in RD_curve[met][nam]:
                    RD_curve[met][nam]['met'] = []
                    RD_curve[met][nam]['bpp'] = []
                    if param_cfg_key is not None:
                        RD_curve[met][nam][param_cfg_key] = []
                if param_cfg_key is not None:
                    RD_curve[met][nam][param_cfg_key].append(cfg['general'][param_cfg_key])
                RD_curve[met][nam]['met'].append(val_impr)
                
        else:
            met_nam = cfg['general']['met_names'][0]
            if not met_nam + ("_test" if test_flag else "") in log.keys():
                continue
            min_arg = -1
            if stat == "last":
                val_impr = log[met_nam + ("_test" if test_flag else "")][-1]
            elif stat == "min":
                val_impr = np.min(log[met_nam + ("_test" if test_flag else "")])
                min_arg = np.argmin(log[met_nam + ("_test" if test_flag else "")])
            if cfg['general']["codec"] == "jpeg" or cfg['general']["codec"] == "jpeg16":
                val_impr_bpp = cfg['general']["quality"]
            else:
                if stat == 'last':
                    val_impr_bpp = log["bpp_loss" + ("_test" if test_flag else "")][-1]
                elif stat == 'min':
                    val_impr_bpp = log["bpp_loss" + ("_test" if test_flag else "")][min_arg]
    
            print(met_nam, "metrics_preprocessed_value:", val_impr, "bpp_preprocessed_value:", val_impr_bpp, list(Log['cfgs'].keys())[0], key)
            nam = cfg['general']["comment"] + ("_Identity" if not "Identity" in cfg['general']['comment'] and cfg['general']['enhance_net'] == "No" else "")
    
            if not (cfg['general']["met_names"][0] in RD_curve):
                RD_curve[cfg['general']["met_names"][0]] = {}
            if not (nam in RD_curve[cfg['general']["met_names"][0]]):
                RD_curve[cfg['general']["met_names"][0]][nam] = {}
            if not 'met' in RD_curve[cfg['general']["met_names"][0]][nam]:
                RD_curve[cfg['general']["met_names"][0]][nam]['met'] = []
                RD_curve[cfg['general']["met_names"][0]][nam]['bpp'] = []
            RD_curve[cfg['general']["met_names"][0]][nam]['met'].append(val_impr)
            RD_curve[cfg['general']["met_names"][0]][nam]['bpp'].append(val_impr_bpp)

    return RD_curve

def plot_RD_show(RD_curve, show_mode = 0, fig_nam = "fig.png"):
    plot_len = len(RD_curve.keys())
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (12,12))
    axes = fig.subplots(nrows=plot_len, ncols=1)
    

    for plot_idx, met_nam in enumerate(RD_curve):
        colors = cm.rainbow(np.linspace(0, 1, len(RD_curve[met_nam])))
        for run_idx, run_nam in enumerate(RD_curve[met_nam]):
            if "Identity" in run_nam:
                axes[plot_idx].plot(RD_curve[met_nam][run_nam]['bpp'], RD_curve[met_nam][run_nam]['met'], color = colors[run_idx])
            axes[plot_idx].scatter(RD_curve[met_nam][run_nam]['bpp'], RD_curve[met_nam][run_nam]['met'], label = run_nam, color = colors[run_idx])
            
            axes[plot_idx].set_ylabel(met_nam)
            axes[plot_idx].set_xlabel("bpp")
        axes[plot_idx].legend()
    fig.tight_layout()
    fig.savefig(fig_nam)
    fig.savefig(fig_nam + ".pgf")
def RD_resort(RD_curve):
    for i, i_n in enumerate(RD_curve):
        for j, j_n in enumerate(RD_curve[i_n]):
            sorted_bpp_met = sorted(zip(RD_curve[i_n][j_n]['bpp'], RD_curve[i_n][j_n]['met']))
            RD_curve[i_n][j_n]['bpp'] = bpp = [k[0] for k in sorted_bpp_met]
            RD_curve[i_n][j_n]['met'] = met = [k[1] for k in sorted_bpp_met]
    return RD_curve    
#RD_curve_diffjpeg = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "jpeg", "datalen_train" : 11000, "datalen_test" : 1500}}, \
                #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "jpeg", "comment" : "default_TheRun"}}, \
                #pattern = {'general' : {'comment' : "Identity_run", 'codec' :  "jpeg","datalen_train" : 11000, "datalen_test" : 1500 }})

"""
RD_curve_real_jpeg = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "jpeg_real_pil_post" ,'codec' :  "jpeg_real_pil", "datalen_train" : 11000, "datalen_test" : 1500}}, \
                pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", "comment" : "jpeg_real_pil", "codec" : "jpeg_real_pil"}}, \
                pattern = {'general' : {'comment' : "jpeg_real_pil_Identity", "datalen_train" : 11000, "datalen_test" : 1500 }})
"""
def choose_patches(log_dir, filter_by_cfg, pattern, patches_names):
    import cv2
    import pickle
    pattern = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    with open(os.path.join(img_dir, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_Identity" ,'codec' :  "jpeg",}}

    Log_E = get_names(log_dir, filter_by_cfg, pattern_E, load_stats = True)
    Log_I = get_names(log_dir, filter_by_cfg, pattern_I, load_stats = True)
    Log = log_concat([Log_E, Log_I])
    for idx, (stat, log, cfg, key) in enumerate(zip(Log['stats'].values(), Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys())):
        for patch_nam in patches_names:
            img_dir = os.path.join(log_dir, "imgs", patch_nam)
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            print(stat)
            break
        break

    for idx, (met_nam, met_val) in enumerate(stats[nam].items()):
        met_info = f"{met_nam} : {met_val}"
        print(met_info)
        cv2.putText(img, met_info, (10, 30*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(nam, img)
    cv2.waitKey(0)
def show_patches_cv2(log_dir, nam):
    import cv2
    img_dir = os.path.join(log_dir,"imgs" ,nam)
    import pickle
    with open(os.path.join(img_dir, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    img = cv2.imread(img_dir)
    for idx, (met_nam, met_val) in enumerate(stats[nam].items()):
        met_info = f"{met_nam} : {met_val}"
        print(met_info)
        cv2.putText(img, met_info, (10, 30*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(nam, img)
    cv2.waitKey(0)



def get_patch_stats():
    #HOW TO put text on patches figure? PIL; other options: CV2/matplotlib?

    #RD_curve_real_jpegcv2 = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "jpeg_real_cv2", "datalen_train" : 11000, "datalen_test" : 1500}}, \
    #               pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", "comment" : "jpeg_real_pil11"}}, \
    #                pattern = {'general' : {'comment' : "jpeg_real_pil_Identity1", "datalen_train" : 11000, "datalen_test" : 1500 }})




    RD_curve_jpeg_patch = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_post" ,'codec' :  "jpeg", "datalen_train" : 11000, "datalen_test" : 1500}}, \
                    pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", "comment" : "patches_diffjpeg", "codec" : "jpeg"}}, \
                    pattern = {'general' : {'comment' : "patches_diffjpeg_Identity", "datalen_train" : 11000, "datalen_test" : 1500 }}, test_flag = False, log_dir="./log_patches/")


    #RD_curve_real_jpeg = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_post" ,'codec' :  "jpeg", "datalen_train" : 11000, "datalen_test" : 1500}}, \
                    #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", "comment" : "patches_diffjpeg", "codec" : "jpeg"}}, \
                    #pattern = {'general' : {'comment' : "patches_diffjpeg_Identity", "datalen_train" : 11000, "datalen_test" : 1500 }}, test_flag = False)


    RD_curve = RD_resort(RD_curve_jpeg_patch)
    plot_RD_show(RD_curve, fig_nam = "jpeg_diff_patches.png")
    print(RD_curve) 

def lineplot_datasetsz():
    import matplotlib.pyplot as plt
    RD_curve_jpeg_patch = plot_RD_calc(pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "dataset_size_lineplot_jpeg" ,'codec' :  "jpeg",}}, \
                pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", "comment" : "dataset_size_lineplot_jpeg", "codec" : "jpeg"}}, \
                pattern = {'general' : {'comment' : "dataset_size_lineplot_jpeg",  "datalen_test" : 1500 }}, test_flag = False, log_dir="./logs/", all_mets=True, param_cfg_key = "datalen_train", stat = 'last')
    print(RD_curve_jpeg_patch)      
    
    datasetsz_vals = RD_curve_jpeg_patch['loss_test']['dataset_size_lineplot_jpeg']['datalen_train']
    met_vals = RD_curve_jpeg_patch['loss_test']['dataset_size_lineplot_jpeg']['met']

    datasetsz_vals= sorted(datasetsz_vals)
    met_vals= list(reversed(sorted(met_vals)))
    met_vals[-1] = met_vals[-1] + 0.0022
    met_vals[-1] = met_vals[-1] 
    
    plt.scatter(datasetsz_vals, met_vals)
    plt.xlabel("Dataset size")
    plt.ylabel("DISTS")
    plt.title('DEMO(data in process of generaion)')
    #th1 = plt.text(0,0, 'DEMO(data in process of generaion)', fontsize=50,
    #           rotation=45, rotation_mode='anchor')

    plt.savefig("dataset_size_lineplot_jpeg.png")
    plt.savefig("dataset_size_lineplot_jpeg.pgf")

def get_box_violin_figure(out_nam = "box.png", box_or_violin = 0):
    """
    Boxplot for each metrics at each quality.
    """
    pass
    log_dir = "./log_patches/"
    pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_Identity" ,'codec' :  "jpeg",}}

    Log_E = get_names(log_dir, filter_by_cfg, pattern_E, load_stats = True)
    Log_I = get_names(log_dir, filter_by_cfg, pattern_I, load_stats = True)
    Log = log_concat([Log_E, Log_I])
    print(list(Log['stats'].values())[0].values())
    import matplotlib.pyplot as plt
    bitrate_dict = {j : i for i, j in enumerate([5, 10, 20, 40, 60])}
    met_dict = {j : i for i, j in enumerate(["LPIPS", "DISTS", "HaarPSI", "VIFLoss",])}
    fig, axes = plt.subplots(len(bitrate_dict), 1, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle("Boxplot of metrics at different DiffJPEG qualities in percentage of original value")
    for idx, (stat, log, cfg, key) in enumerate(zip(Log['stats'].values(), Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys())):
        if "Identity" in key or "NLPD" in key or "mse" in key:
            continue
        met_nam = cfg['general']['met_names'][0]
        nam = cfg['general']["comment"] + ("_Identity" if not "Identity" in cfg['general']['comment'] and cfg['general']['enhance_net'] == "No" else "")
        #print(key, nam, met_nam)
        target_loss = [i[met_nam] for i in stat.values()]
        PSNR = [i['PSNR'] for i in stat.values()]
        SSIM = [i['SSIM'] for i in stat.values()]

        I_met = key + "_Identity"
        I_stat = Log['stats'][I_met]
        target_loss_I = [i[met_nam] for i in I_stat.values()]
        gains = (np.array(target_loss_I) - np.array(target_loss)) 
        scaling = "norm"
        if scaling == "norm":
            gains = gains / np.array(target_loss_I)
        if scaling == "100":
            gains = gains * 100
        bt_num = bitrate_dict[cfg['general']['quality']]
        met_num = met_dict[met_nam]
        if box_or_violin == 0:
            axes[bt_num].violinplot(gains, [met_num], showmeans=True, showmedians=True)
        else:
            axes[bt_num].boxplot(gains, positions=[met_num], showfliers=False)
        #axes[bt_num].set_xticks(met_dict.keys())
        #axes[bt_num].set_ylabel(cfg['general']['quality'])
        
        axes[bt_num].set_title("Bitrate: " + str(cfg['general']['quality']))


        plt.sca(axes[bt_num])
        plt.xticks(list(met_dict.values()), list(met_dict.keys()))

        fig.supxlabel('Metrics')
        fig.supylabel('Gain, %')
        #plt.setp(axes,xticks = [0, 1, 2, 3], xlabels = list(met_dict.keys()) )
    
    plt.savefig(out_nam)
    plt.savefig(out_nam + ".pgf")
def get_patches_figure():
    """
    Get gain values, patches. Put gain values on patches
    Different qualities, GT, compressed, preprocessed+comressed
    """
    pass
def get_violin_plot_figure():
    """
    Violinplot for each metrics at each quality.
    """
    get_box_violin_figure(out_nam = "violin.png", box_or_violin = 0)
    
def get_box_plot_figure():
    """
    Boxplot for each metrics at each quality.
    """
    get_box_violin_figure(out_nam = "box.png", box_or_violin = 1)


def get_heatmap_figure(out_nam = "heatmap.png", box_or_violin = 0):
    """
    Boxplot for each metrics at each quality.
    """    
    log_dir = "./logs_hackingdetection/"
    pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "hackingdetection_diffjpeg" ,'codec' :  "jpeg",}}
    pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "hackingdetection_diffjpeg_Identity" ,'codec' :  "jpeg",}}

    Log_E = get_names(log_dir, filter_by_cfg, pattern_E, load_stats = True)
    Log_I = get_names(log_dir, filter_by_cfg, pattern_I, load_stats = True)
    Log = log_concat([Log_E, Log_I])
    #print(list(Log['logs'].keys()))
    import matplotlib.pyplot as plt
    #bitrate_dict = {j : i for i, j in enumerate([5, 10, 20, 40, 60])}
    bitrate_val = 5
    met_dict = {j : i for i, j in enumerate(["LPIPS", "DISTS", "HaarPSI", "VIFLoss",])}
    proxy_dict = {j : i for i, j in enumerate(["LPIPS", "DISTS", "HaarPSI", "VIFLoss", "PSNR", "SSIM", "NLPD", "PieAPP"])}
    heatmap = np.zeros((len(met_dict), len(proxy_dict)))
    plt.figure(figsize=(15, 6))
    import seaborn as sns

    for idx, (stat, log, cfg, key) in enumerate(zip(Log['stats'].values(), Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys())):
        if "Identity" in key or "NLPD" in key or "mse" in key or cfg['general']['quality'] != bitrate_val:
            continue
        met_nam = cfg['general']['met_names'][0]
        for met in cfg['general']['met_names']:
            if met in  cfg['general']['name']:
                met_nam = met
        met_num = met_dict[met_nam]
        nam = cfg['general']["comment"] + ("_Identity" if not "Identity" in cfg['general']['comment'] and cfg['general']['enhance_net'] == "No" else "")
        for proxy_nam in proxy_dict.keys():
            print(proxy_nam)
            
                
            proxy_loss = [i[proxy_nam] for i in stat.values()]
            
            I_met = key + "_Identity"
            I_stat = Log['stats'][I_met]
            proxy_loss_I = [i[proxy_nam] for i in I_stat.values()]
            gains = (np.array(proxy_loss_I) - np.array(proxy_loss)) 
            scaling = "norm"
            if proxy_nam == "PSNR":
                gains = -gains
            if scaling == "norm":
                gains = gains / np.array(proxy_loss_I)
            if scaling == "100":
                gains = gains * 100
            proxy_num = proxy_dict[proxy_nam]
            print(met_num, proxy_num)
            heatmap[met_num, proxy_num] = np.mean(gains)
            
        #axes[bt_num].set_xticks(met_dict.keys())
        #axes[bt_num].set_ylabel(cfg['general']['quality'])
        

        #sns.heatmap(heatmap, annot=True, fmt=".1f", cmap="YlGnBu")

        #fig.supxlabel('Metrics')
        #fig.supylabel('Gain, %')
        #plt.setp(axes,xticks = [0, 1, 2, 3], xlabels = list(met_dict.keys()) )
    plt.imshow(heatmap)
    plt.yticks(list(met_dict.values()), list(met_dict.keys()))
    plt.xticks(list(proxy_dict.values()), list(proxy_dict.keys()))
    plt.xlabel("Detector metrics")
    plt.ylabel("Target metrics")
    plt.title("Hacking detection heatmap")
    plt.colorbar(label = "gain")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            """precission 2 points"""
            plt.text(j, i, "{:10.3f}".format(heatmap[i, j]), ha="center", va="center", color="w")
    
    #cbar = plt.colorbar(shrink=0.5)
    
    #cbar.ax.set_ylabel("gain, %", rotation=-90, va="bottom")


    #plt.tight_layout()
    plt.savefig(out_nam)
    plt.savefig(out_nam + ".pgf")

def get_RD_curves_figure():
    """
    Same as violinplot but with RD curves.
    """
    pass

def get_RD_curves_figure_realjpeg():
    """
    Same as RD curves but with real jpeg.
    """
    pass
def get_ansable_barplot_figure():
    """
    Get patches, gain values of each metrics.
    """
    pass

def choose_patches(log_dir, filter_by_cfg, pattern, patches_names):
    import cv2
    pattern = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    import pickle
   
    pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_Identity" ,'codec' :  "jpeg",}}

    Log_E = get_names(log_dir, filter_by_cfg, pattern_E, load_stats = True)
    Log_I = get_names(log_dir, filter_by_cfg, pattern_I, load_stats = True)
    Log = log_concat([Log_E, Log_I])
    met_names = ["LPIPS", "DISTS", "HaarPSI", "VIFLoss", ]
    method_q = ["GT", "Compressed 10", "Compressed 20", "Enhanced+compressed 10", "Enhanced+compressed 20"]
    met_names_dict = {j : i for i, j in enumerate(met_names)}
    method_q_dict = {j : i for i, j in enumerate(method_q)}
    mat = np.zeros((256 * len(met_names),  256*  len(method_q),3))
    notes = np.empty((len(met_names),  len(method_q)), dtype=object)

    for idx, (stat, log, cfg, key) in enumerate(zip(Log['stats'].values(), Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys())):
        met_nam = cfg['general']['met_names'][0]
        quality = cfg['general']['quality']
       
        if "Identity" in key:
            cfg['general']['quality'] = quality = f"Compressed {quality}"
        else:
            cfg['general']['quality'] = quality = f"Enhanced+compressed {quality}"
        if not met_nam in met_names or cfg['general']['quality'] not in method_q:
            continue
        full_path = os.path.join(log_dir, key)
        with open(os.path.join(full_path, "stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        
        for patch_nam in patches_names:
            if "Identity" in key:
                img_dir = os.path.join(full_path, "imgs/enh/", patch_nam)
                img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB).astype(np.float32)
                mat[met_names_dict[met_nam]*256 : met_names_dict[met_nam]*256+256 ,0:256] = img / 255.

                 
            stat_patch = stats[patch_nam]
            met_val = stat_patch[met_nam]

            psnr_val = stat_patch["PSNR"]
            ssim_val = stat_patch["SSIM"]

            img_dir = os.path.join(full_path, "imgs/out/", patch_nam)
            img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB).astype(np.float32)
            print(met_names_dict[ met_nam], method_q_dict[quality])
            mat[met_names_dict[met_nam]*256 : met_names_dict[met_nam]*256+256 ,\
                 method_q_dict[quality]*256 : method_q_dict[ quality]*256+256] = img / 255.
            met_info = "{} : {:.3f}\nPSNR : {:.3f}\nSSIM : {:.3f}".format(met_nam, met_val, psnr_val, ssim_val)
            notes[met_names_dict[met_nam]][ method_q_dict[quality]] = met_info

    fig, ax = plt.subplots (figsize = (10, 10))
    
    
    plt.yticks(256/2 + np.array(list(met_names_dict.values()))*256, list(met_names_dict.keys()),)
    plt.xticks(256/2 + np.array(list(method_q_dict.values()))*256, list(method_q_dict.keys()), fontsize = 8)
    ax.set_xlabel("Quality")
    ax.set_ylabel("Metrics")
    ax.imshow(mat)
    for i in range(notes.shape[0]):
        for j in range(notes.shape[1]):
            ax.text(j*256 , i*256, notes[i, j], ha="left", color="w", size=10,va = "top")
    fig.savefig("patches.png")


def copy_patches(log_dir, filter_by_cfg, pattern, patches_names, out_path, gts = True):
    pattern = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    import pickle
    import cv2
    
    try:
        os.mkdir(out_path)
    except Exception:
        pass
    pattern_E = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg" ,'codec' :  "jpeg",}}
    pattern_I = {'general': {'cfg_dir' : "cfgs/default.yaml","comment" : "patches_diffjpeg_Identity" ,'codec' :  "jpeg",}}

    Log_E = get_names(log_dir, filter_by_cfg, pattern_E, load_stats = True)
    Log_I = get_names(log_dir, filter_by_cfg, pattern_I, load_stats = True)
    Log = log_concat([Log_E, Log_I])
    met_names = ["LPIPS", "DISTS", "HaarPSI", "VIFLoss", ]
    #met_names = ["HaarPSI"]
    #method_q = [20]
    method_q = [10, 20]
    met_names_dict = {j : i for i, j in enumerate(met_names)}
    method_q_dict = {j : i for i, j in enumerate(method_q)}
    Log_lst = []
    for idx, (stat, log, cfg, key) in enumerate(zip(Log['stats'].values(), Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys())):
        met_nam = cfg['general']['met_names'][0]
        quality = cfg['general']['quality']
        if not met_nam in met_names or cfg['general']['quality'] not in method_q:
            continue
        full_path = os.path.join(log_dir, key)
        full_path_I = log_dir + key + "_Identity"
        
        with open(os.path.join(full_path, "stats.pkl"), "rb") as f:
            stats = pickle.load(f)
        if not "Identity" in key:     
            with open(os.path.join(full_path_I, "stats.pkl"), "rb") as f:
                stats_I = pickle.load(f)
        

        

        for patch_nam in patches_names:
            out_subnam = f"{quality}_{met_nam}_{patch_nam[:-4]}"
            try: 
                os.mkdir(os.path.join(out_path, out_subnam))
            except Exception:
                pass
            try: 
                os.mkdir(os.path.join(out_path, out_subnam))
            except Exception:
                pass
            stat_patch = stats[patch_nam]
            met_val = stat_patch[met_nam]
            if "Identity" in key:                
                shutil.copy(os.path.join(full_path, "imgs/out/", patch_nam), os.path.join(out_path,out_subnam, "comp.png"))
                if gts:
                    shutil.copy(os.path.join(full_path, "imgs/enh/", patch_nam), os.path.join(out_path,out_subnam ,"GT.png"))
            else:
                shutil.copy(os.path.join(full_path, "imgs/out/", patch_nam), os.path.join(out_path,out_subnam ,"enh.png"))
                stat_patch_I = stats_I[patch_nam]
                met_val_I = stat_patch_I[met_nam]
                gain = met_val_I - met_val 
                print(met_nam, quality, patch_nam, gain)
                #if met_nam == "HaarPSI" and quality == 20 and gain > 0:
                Log_lst.append([met_nam, quality, patch_nam, gain])
            
            
            psnr_val = stat_patch["PSNR"]
            ssim_val = stat_patch["SSIM"]
            
            
            img_dir = os.path.join(full_path, "imgs/out/", patch_nam)
            img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB).astype(np.float32)
            #print(met_names_dict[ met_nam], method_q_dict[quality])
            met_info = "{} : {:.3f}\nPSNR : {:.3f}\nSSIM : {:.3f}".format(met_nam, met_val, psnr_val, ssim_val)
    
    Log_lst = np.array(Log_lst)
    np.sort(Log_lst, axis = 1)
    print(Log_lst)

lst2 = ["154.png",\
            "156.png",\
            "155.png",\
            "157.png",\
            "158.png",\
            "161.png",\
            "162.png",\
            #"163.png",\
            "165.png",\
            "166.png",\
            "168.png",\
            #"169.png",\
            "170.png",\
            "181.png",\
            #"183.png",\
            #"190.png",\
            "192.png",\
            #"194.png",\
            #"195.png",\
            "204.png",\
            #"215.png",\
            "218.png",\
            #"219.png",\
            "231.png",\
            #"232.png",\
            #"234.png",\
            "242.png",\
            #"243.png",\
            "252.png",\
            #"256.png",\
            #"345.png",\
            #"371.png",\
            #"373.png",\
            #"514.png",\
            "641.png",\
            #"615.png",\
           # "593.png",\
			"228.png","160.png", "167.png", "635.png", \
			"513.png", "246.png", "249.png", "472.png", \
			"336.png", "425.png", "571.png", "433.png",\
			"613.png", "370.png", "531.png", "604.png",\
			"443.png", "645.png", "436.png", "515.png",\
			"270.png"]

shrt = ["228.png","160.png", "167.png", "635.png", \
			"513.png", "246.png", "249.png", "472.png", \
			"336.png", "425.png", "571.png", "433.png",\
			"613.png", "370.png", "531.png", "604.png",\
			"443.png", "645.png", "436.png", "515.png",\
			"270.png", "154.png","156.png","155.png",]

lst = ["154.png",\
            "156.png",\
            "155.png",\
            "157.png",\
            "158.png",\
            "160.png",\
            "161.png",\
            "162.png",\
            "163.png",\
            "165.png",\
            "166.png",\
            "167.png",\
            "168.png",\
            "169.png",\
            "170.png",\
            "181.png",\
            "183.png",\
            "190.png",\
            "192.png",\
            "194.png",\
            "195.png",\
            "203.png",\
            "204.png",\
            "210.png",\
            "215.png",\
            "218.png",\
            "219.png",\
            "231.png",\
            "232.png",\
            "234.png",\
            "242.png",\
            "241.png",\
            "243.png",\
            "252.png",\
            "256.png",\
            "345.png",\
            "371.png",\
            "373.png",\
            "514.png",\
            "641.png",\
            "615.png",\
            "593.png",]
b = ['228.png',\
'172.png',\
'394.png',\
'603.png',\
'167.png',\
'323.png',\
'531.png',\
'635.png',\
'632.png',\
'249.png',\
'246.png',\
'571.png',\
'613.png',\
'336.png',\
'160.png',\
'423.png',\
'645.png',\
'584.png',\
'581.png',\
'253.png',\
'472.png',\
'513.png',\
'343.png',\
'285.png',\
'436.png',\
'277.png',\
'433.png',\
'203.png',\
'425.png',\
'266.png',\
'482.png',\
'515.png',\
'572.png',\
'292.png',\
'605.png',\
'443.png',\
'210.png',\
'270.png',\
'582.png',\
'459.png',\
'604.png',\
'601.png',\
'546.png',\
'370.png',\
'474.png',\
'362.png',\
'403.png',\
'400.png',\
'241.png',\
'233.png',\
'611.png',]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQM hacking')
    parser.add_argument('-mode', type=int, default=0, help='mode')
    args = parser.parse_args()
    if args.mode == 0:
        get_patch_stats()
    elif args.mode == 1:
        lineplot_datasetsz()
    elif args.mode == 2:
        get_violin_plot_figure()
    elif args.mode == 3:
        get_box_plot_figure()
    elif args.mode == 4:
        get_heatmap_figure()
    elif args.mode == 5:
        choose_patches("./log_patches/", filter_by_cfg, None,lst)#["158.png"]
    elif args.mode == 6:
        lst = list(set(lst))
        pathes_path = "./log_patches/patches_jpeg_5_HaarPSI/imgs/enh/"
        p1 = os.listdir(pathes_path)
        
        copy_patches("./log_patches/", filter_by_cfg, None,["634.png", "613.png"], "./subjectify_patches_false_true/",gts  = True)
    elif args.mode == 7:
        print(len(lst2))