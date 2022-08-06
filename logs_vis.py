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
    pattern1 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun2107"}}
    pattern2 = {'general': {'cfg_dir' : "cfgs/default.yaml", 'codec' :  "cheng2020_attn", "comment" : "default_TheRun"}}

    Log = get_names("./logs/", filter_by_cfg, pattern2)
    Log1 = get_names("./logs/", filter_by_cfg, pattern1)
    Log = log_concat([Log, Log1])
    for log, cfg, key in zip(Log['logs'].values(), Log['cfgs'].values(), Log['cfgs'].keys()):
        pattern = {'general' : {'enhance_net' : "No", "codec" : cfg['general']['codec'],  'met_names' :  cfg['general']["met_names"], "quality" : cfg['general']["quality"], "sigmoid_activation" : 0}}#"codec_metric" : cfg['general']["codec_metric"], "quality" : cfg['general']["quality"]
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
            print(met_nam,"met", val_no - val_impr, "bpp",val_no_bpp - val_impr_bpp,list(Log_I['cfgs'].keys())[0], key)
plot_all()
