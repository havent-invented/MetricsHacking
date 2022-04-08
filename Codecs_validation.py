tmp = None
cfg["run"]["device"] = 'cpu'
cfg["run"]["device_met"] = "cuda:0"
cfg["run"]["device_enh"] = "cuda:0"
cfg["general"]["minimal_batch_sz"] = 2
cfg["run"]["device_sub"] = "cpu"
cfg["general"]["home_dir"] = "R:/home_dir/"
cfg["general"]["batch_size_test"] = 1#4
cfg["general"]["num_frames"] = 64
dst_dir = "P:/7videos/"
dst_dir_vimeo = 'P:/vimeo_triplet/sequences/'
exec(open('main.py').read())

try:
    os.mkdir(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))
except Exception:
    pass

X_out1 = None
import compressai
import math
from compressai.zoo import bmshj2018_factorized, cheng2020_attn, mbt2018,ssf2020
import torch
from PIL import Image
import torchvision.transforms
import torch
import skvideo.io
from PIL import Image
import numpy as np
from CNNfeatures import get_features
from VQAmodel import VQAModel
from argparse import ArgumentParser
import time
from PIL import Image
import torch
import numpy as np
from torch import nn
import torch.optim as optim

from torch.utils.data import Dataset, IterableDataset

from torchvision.io import read_image
from torch.utils.data import DataLoader
import os
import torchvision


net_enhance = None

def load_models(paths):
    import os
    model_list = []
    for path in paths:
        if os.path.getsize(path) // 10**6 == 73:
            model_list.append(ResNetUNet(3))
        if os.path.getsize(path) // 10**6 == 10:
            model_list.append(get_simple_cnn())
        model_list[-1].load_state_dict(torch.load(path)) 
    return model_list

def compute_model_codec_dataset(net_enhance, net_codec, dataset, cfg, Y_dataset = None, vid_full_dir = None):
    logs_plot_cur = {}
    
    #new code
    if vid_full_dir != None:
        dataset = Video_reader_dataset(name1 = vid_full_dir, num_frames = cfg["general"]["num_frames"], minimal_batch_sz = cfg["general"]["minimal_batch_sz"])   
        dataset = DataLoader(dataset, batch_size= cfg["general"]["batch_size_test"], shuffle = False)

    
    net_codec, cfg["run"]["loss_calc"] = net_codec.to(cfg["run"]["device_sub"]),  cfg["run"]["loss_calc"].to(cfg["run"]["device_sub"])
    XY_flag = False
    if Y_dataset != None:
        frame_XY_all = (dataset, Y_dataset)
    else:
        frame_XY_all = (dataset, range(len(dataset)))
    for XY_frame in tqdm(zip(*frame_XY_all)):
        X = XY_frame[0]
        if cfg["general"]["to_crop"]:
            X = torchvision.transforms.CenterCrop((cfg["general"]["patch_sz"], cfg["general"]["patch_sz"]))(X)
        X = X.detach().to(cfg["run"]["device_met"])
        if Y_dataset == None:
            Y = X.detach().clone().to(cfg["run"]["device_met"])
        else:
            Y = XY_frame[1].to(cfg["run"]["device_met"])
        if cfg["general"]["to_crop"]:
            Y = torchvision.transforms.CenterCrop((cfg["general"]["patch_sz"], cfg["general"]["patch_sz"]))(Y)
        X, net_enhance = X.to(cfg["run"]["device_enh"]), net_enhance.to(cfg["run"]["device_enh"])
        X_enhance = net_enhance(X)
        X, net_enhance = X.to(cfg["run"]["device_sub"]), net_enhance.to(cfg["run"]["device_sub"])
        torch.cuda.empty_cache()
        X_enhance, net_codec = X_enhance.to(cfg["run"]["device_met"]), net_codec.to(cfg["run"]["device_met"])
        X_out = net_codec(X_enhance)
        X_out["x_hat"] = X_out["x_hat"][..., :X_enhance.shape[-2], :X_enhance.shape[-1]]
        net_codec = net_codec.to(cfg["run"]["device_sub"])
        torch.cuda.empty_cache()
        
        cfg["run"]["loss_calc"] = cfg["run"]["loss_calc"].to(cfg["run"]["device_met"])
        Y = Y.to(cfg["run"]["device_met"])
        cfg["run"]["tmp"] = X_enhance, X_out['x_hat'], Y
        loss = cfg["run"]["loss_calc"](X_out, Y)
        cfg["run"]["loss_calc"] = cfg["run"]["loss_calc"].to(cfg["run"]["device_sub"])
        
        for j in list(loss.keys()):
            if not j in logs_plot_cur:
                logs_plot_cur[j] = []
            logs_plot_cur[j].append(loss[j].data.to(cfg["run"]["device_sub"]).numpy())
        X.data.clamp_(min=0,max=1)
        X_out['x_hat'].data.clamp_(min=0,max=1)
    for j in list(logs_plot_cur.keys()):
        logs_plot_cur[j] = np.mean(logs_plot_cur[j])
    if vid_full_dir != None:        
        dataset.dataset.close()
        del dataset
    return logs_plot_cur

def append_dict(dict_from, dict_to):
    for j in list(dict_from.keys()):
        if not j in dict_to:
            dict_to[j] = []
        dict_to[j].append(np.mean(dict_from[j]))    
    return dict_to
    
def model_codecs_dataset(net_enhance, net_codecs, dataset, cfg, vid_full_dir = None):
    logs_plot = {}
    for net_codec in net_codecs:
        net_codec_gpu = net_codec.to(cfg["run"]["device_met"])
        net_enhance_gpu = net_enhance.to(cfg["run"]["device_met"])
        logs_plot_cur = compute_model_codec_dataset(net_enhance_gpu, net_codec_gpu, dataset, cfg, vid_full_dir = vid_full_dir)
        logs_plot = append_dict(logs_plot_cur, logs_plot)
        del net_codec_gpu
        del net_enhance_gpu 
    return logs_plot
    
def models_codecs_dataset(net_enhances, net_codecs, dataset, cfg, vid_full_dir = None):
    logs_plot = []
    for net_enhance in net_enhances:
        logs_plot_cur = model_codecs_dataset(net_enhance, net_codecs, dataset, cfg, vid_full_dir = vid_full_dir)
        logs_plot.append(logs_plot_cur)
    return logs_plot    

def compare_models(models, net_codec, dataset):
    import os
    log_all = []
    with torch.no_grad():
        logs_plot = {}
        for model in models:
            logs_plot_cur = compute_model_codec_dataset(model, net_codec, dataset, cfg)
            for j in list(logs_plot_cur.keys()):
                if not j in logs_plot:
                    logs_plot[j] = []
                logs_plot[j].append(np.mean(logs_plot_cur[j]))
            log_all.append(logs_plot)
    return log_all

#on videos
class codec_outer_raw():
    def __init__(self, device = cfg["run"]["device"], home_dir = cfg["general"]["home_dir"], output_dir = None, codec = None):
        import pickle
        self.codec = codec
        self.device = device
        self.home_dir = home_dir
        self.out1 = None
        if output_dir == None:
            self.output_dir = self.home_dir + "0YES.Y4M"
        else:
            self.output_dir = output_dir
    def forward(self, X):
        if self.out1 == None:
            self.out1 = skvideo.io.FFmpegWriter(self.output_dir ,inputdict = {"-pix_fmt": "rgb24"}, outputdict = {"-pix_fmt": "yuv420p"})
        for img in X.to(cfg["run"]["device_sub"]).detach().numpy().swapaxes(1,3).swapaxes(1,2):
            if img.max() < 2.:
                img = img * 255
            self.out1.writeFrame(img)
        return {"x_hat": X}
    def close(self):
        if self.out1 != None:
            self.out1.close()
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self


class codec_outer_compress:
    def __init__(self, home_dir, codec, input_dir = None, compressed_dir = None, output_dir = None):
        self.home_dir = home_dir
        self.bitrate = 0
        if compressed_dir == None:
            self.compressed_dir = home_dir + "/a.mp4"
        else:
            self.compressed_dir = compressed_dir
        if output_dir == None:
            self.output_dir = home_dir + "/0YES.Y4M"
        else:
            self.output_dir = output_dir
        if input_dir == None:
            self.input_dir = home_dir + "/0YES.Y4M"
        else:
            self.input_dir = input_dir
        self.codec = codec#' -preset:v medium -x265-params log-level=error '
    def forward(self):#-c:v mjpeg 
        os.system("ffmpeg -hide_banner -loglevel error -y -i " + self.input_dir + " " + self.codec + "  -pix_fmt yuv420p " + self.compressed_dir)
        os.system("ffmpeg -hide_banner -loglevel error -y -i " + self.input_dir + " " + self.codec + "  -pix_fmt yuv420p  " + self.compressed_dir)
        os.system("ffmpeg -hide_banner -loglevel error -y -i " + self.compressed_dir + " -pix_fmt yuv420p " + self.output_dir)
        self.bitrate = int(skvideo.io.ffprobe(self.compressed_dir)['video']['@bit_rate']) / 10**6
    def get_bitrate(self):
        return self.bitrate
    def __call__(self):
        return self.forward()
   

        
def compute_model_codec_dataset_outer(net_enhance_gpu, cfg, to_enhance = True, vid_full_dir = None, dataset = None, 
                                      codec = None, bitrates = None,home_dir = cfg["general"]["home_dir"],  ):
    #try:
    #    num_frames
    #except Exception:
        #num_frames = 64
    if to_enhance:
        #if dataset == None:
        dataset_test = Video_reader_dataset(name1 = vid_full_dir, num_frames = cfg["general"]["num_frames"], minimal_batch_sz = cfg["general"]["minimal_batch_sz"])   
        dataset_test = DataLoader(dataset_test, batch_size= cfg["general"]["batch_size_test"], shuffle = False)
        codec_raw = codec_outer_raw(device = cfg["run"]["device"], codec = codec, 
                                    output_dir = os.path.join(home_dir, "0YES.Y4M"))
        compute_model_codec_dataset(net_enhance_gpu, codec_raw, dataset_test, cfg)
        codec_raw.close()
        name1 = dataset_test.dataset.nameGT
        datalen1 = dataset_test.dataset.datalen
        batch_size1 = dataset_test.batch_size
        dataset_test.dataset.close()
        del dataset_test
        dataset_test = None
    dataset_test = Video_reader_dataset(name1 = vid_full_dir, num_frames = cfg["general"]["num_frames"], minimal_batch_sz = cfg["general"]["minimal_batch_sz"])
    dataset_test = DataLoader(dataset_test, batch_size= cfg["general"]["batch_size_test"], shuffle = False)
    codec_compressor = codec_outer_compress(home_dir = home_dir,
                         codec = codec, input_dir = os.path.join(home_dir, "0YES.Y4M"), 
                                            output_dir = os.path.join(home_dir, "0YES_comp.Y4M"))
    codec_compressor()
    dataset_test_comp = Video_reader_dataset(name1 = os.path.join(home_dir, "0YES_comp.Y4M"), num_frames = cfg["general"]["num_frames"])
    dataset_test_comp = DataLoader(dataset_test_comp, batch_size= cfg["general"]["batch_size_test"], shuffle = False)
    logs_plot_cur = compute_model_codec_dataset(enhance_Identity, codec_Identity, dataset_test_comp, 
                                                cfg, Y_dataset = dataset_test)
    logs_plot_cur['bitrate'] = codec_compressor.get_bitrate()
    dataset_test.dataset.close()
    dataset_test_comp.dataset.close()
    return logs_plot_cur
    
def model_codecs_dataset_outer(net_enhance, cfg, vid_full_dir = None,dataset = None, codecs = None, 
                               bitrates = None,home_dir = cfg["general"]["home_dir"], ):
    if codecs == None:
        raise Exception
    logs_plot = {}
    to_enhance = True
    for codec in codecs:
        if to_enhance:
            net_enhance_gpu = net_enhance.to(cfg["run"]["device_met"])
        else:
            net_enhance_gpu = None
        logs_plot_cur = compute_model_codec_dataset_outer(net_enhance_gpu, cfg, to_enhance = to_enhance,codec = codec, vid_full_dir = vid_full_dir, dataset = None)###
        to_enhance = True#False
        logs_plot = append_dict(logs_plot_cur, logs_plot)
        del net_enhance_gpu 
    return logs_plot
    
def models_codecs_dataset_outer(net_enhances, cfg, vid_full_dir = None,  dataset = None, codecs = None,
                                ):
    if type(codecs) == list and type(codecs[0]) != str:
        return models_codecs_dataset(net_enhances, codecs, dataset, cfg, vid_full_dir)
    logs_plot = []
    for net_enhance in net_enhances:
        logs_plot_cur = model_codecs_dataset_outer(net_enhance,cfg, codecs = codecs, dataset = None, vid_full_dir=vid_full_dir)
        logs_plot.append(logs_plot_cur)
    return logs_plot
  

def get_met_names(directory = "./models_enhancement/", key = "fixed_direction", force_names = None):
    if force_names != None:
        return force_names, force_names, force_names
    if type(key) == str:
        model_dir_full = sorted([os.path.join(directory, i) for i in os.listdir(directory) if key in i])
    else:
        model_dir_full = sorted([os.path.join(directory, i) for i in os.listdir(directory) if key(i)])
    model_target_met_name = list(map(lambda x : x.split("_")[3], model_dir_full))
    model_names = [model_name.split("models_enhancement/model_vimeo11k_")[-1].split(".ckpt")[0] for model_name in model_dir_full ]
    return model_dir_full, model_target_met_name,model_names
import matplotlib
import matplotlib.cm as cm

def RD_curves_plot(test_RDcurves, videoname = sorted(os.listdir(dst_dir))[0], save_pgf = False, save_png = False, fig_file = "./vis/RD_curves/", force_names_all = None):        
    import matplotlib.pyplot as plt
    if force_names_all != None:
        model_dirs_full, model_target_met_names, model_names = force_names_all
    else:
        model_dirs_full, model_target_met_names, model_names = get_met_names(key='fixed_direction')
    met_num_table = {i:idx for idx,i in enumerate(set(model_target_met_names))}
    count_Original_table = {i:0 for idx,i in enumerate(set(model_target_met_names))}
    met_num_max = len(met_num_table)
    fig, plt_sub = plt.subplots(2,met_num_max,figsize = (40,10), facecolor=(1, 1, 1))
    plt_sub = plt_sub.T.ravel()
    for i in plt_sub:
        i.grid()
    fig.suptitle(videoname)
    for (idx, j), name,target_met in zip(enumerate(test_RDcurves), model_names, model_target_met_names):
        if name != target_met:
            p = name.split("_")[:2] + [name.split("quality")[-1][:1]]
            if len(p) <= 2:
                p = name + " no preprocessing"
            else:
                p = p[0] +"+"+ p[1] + (" tuned for quality " + p[2] if p[2].isdigit() else " tuned without codec")
        else:
            p = name + " no preprocessing"

        print(met_num_table, target_met,met_num_table[target_met])
        plt_sub[2*met_num_table[target_met]].plot(j[0]['bitrate']     [:len(j[0]['mse'])], 10 * np.log10(1. / np.array(j[0]['mse'])), label = p)
        plt_sub[2*met_num_table[target_met]].set_ylabel("PSNR")
        plt_sub[2*met_num_table[target_met]].set_xlabel("bitrate")
        plt_sub[2*met_num_table[target_met] + 1].plot(j[0]['bitrate']   [:len(j[0]['mse'])], -np.array(j[0][target_met]), label = p)
        plt_sub[2*met_num_table[target_met] + 1].set_ylabel(target_met)
        plt_sub[2*met_num_table[target_met] + 1].set_xlabel("bitrate")
        
        #if count_Original_table[target_met] == 0:
        #    count_Original_table[target_met] += 1
            #plt_sub[2*met_num_table[target_met]].plot(j[1]['bitrate'], 10 * np.log10(1. / np.array(j[1]['mse_loss'])), label = "Original")
            #plt_sub[2*met_num_table[target_met] + 1].plot(j[1]['bitrate'], -np.array(j[1][target_met]), label = "Original")
        plt_sub[2*met_num_table[target_met] + 1].legend()
        plt_sub[2*met_num_table[target_met]].legend()
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.tight_layout() 
    if save_png:
        plt.savefig(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "RDcurves.png"),bbox_inches='tight')
    if save_pgf:
        plt.savefig(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "RDcurves.pgf"),bbox_inches='tight')