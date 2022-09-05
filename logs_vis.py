from cProfile import label
import numpy as np
import os
import pickle
import yaml
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

def get_names(log_dir, f_func, pattern):
    logs = {"cfgs": {}, "logs" : {}}
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
        idx += 1
    return logs


#pattern3 =  {'general': {'enhance_net' :  "No", }}
#A = get_names("./logs/", filter_by_cfg, pattern3)
#met_nam = A['cfgs'][list(A['cfgs'].keys())[0]]['general']['met_names'][0]
#print(A['logs'][list(A['cfgs'].keys())[0]][met_nam + "_test"][-1])
def log_concat(cfgs):
    cfg_res = {"cfgs": {}, "logs": {}}
    idx = 0
    for cfg1 in cfgs:
        for key, cfg2_cfgs, cfg2_logs in zip(cfg1['cfgs'].keys(), cfg1['cfgs'].values(), cfg1['logs'].values()):
            cfg_res['cfgs'][key] = cfg2_cfgs
            cfg_res['logs'][key] = cfg2_logs
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
#plot_all()

def plot_RD_calc():
    #pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun2107"}}
    #pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun"}}
    pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "datalen_train" : 11000, "datalen_test" : 1500}}
    pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "Blur1", "comment" : "default_TheRun"}}
    pattern = {'general' : {'comment' : "Identity_run", "datalen_train" : 11000, "datalen_test" : 1500 }}#"codec_metric" : cfg['general']["codec_metric"], "quality" : cfg['general']["quality"]
    # 'met_names' :  [cfg['general']["met_names"][0]]
    Log = get_names("./logs/", filter_by_cfg, pattern2)
    Log1 = get_names("./logs/", filter_by_cfg, pattern1)
    Log_I =  get_names("./logs/", filter_by_cfg, pattern)
    Log = log_concat([Log, Log1, Log_I])
    RD_curve = {}
    #
    for log, cfg, key in zip(Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys()):
        met_nam = cfg['general']['met_names'][0]
        val_impr = log[met_nam + "_test"][-1]
        val_impr_bpp = log["bpp_loss_test"][-1]

        print(met_nam, "metrics_preprocessed_value:", val_impr, "bpp_preprocessed_value:", val_impr_bpp, list(Log['cfgs'].keys())[0], key)
        nam = cfg['general']["comment"]
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

def plot_RD_show(RD_curve, show_mode = 0):
    plot_len = len(RD_curve.keys())
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (12,12))
    axes = fig.subplots(nrows=plot_len, ncols=1)
    for plot_idx, met_nam in enumerate(RD_curve):
        for run_idx, run_nam in enumerate(RD_curve[met_nam]):
            axes[plot_idx].plot(RD_curve[met_nam][run_nam]['bpp'], RD_curve[met_nam][run_nam]['met'], label = run_nam)
            axes[plot_idx].scatter(RD_curve[met_nam][run_nam]['bpp'], RD_curve[met_nam][run_nam]['met'])
            axes[plot_idx].set_ylabel(met_nam)
            axes[plot_idx].set_xlabel("bpp")
        axes[plot_idx].legend()
    fig.tight_layout()
    fig.savefig("fig.png")
def RD_resort(RD_curve):
    for i, i_n in enumerate(RD_curve):
        for j, j_n in enumerate(RD_curve[i_n]):
            sorted_bpp_met = sorted(zip(RD_curve[i_n][j_n]['bpp'], RD_curve[i_n][j_n]['met']))
            RD_curve[i_n][j_n]['bpp'] = bpp = [k[0] for k in sorted_bpp_met]
            RD_curve[i_n][j_n]['met'] = met = [k[1] for k in sorted_bpp_met]
    return RD_curve    
RD_curve = plot_RD_calc()
RD_curve = RD_resort(RD_curve)
plot_RD_show(RD_curve)
print(RD_curve)