import sys
import skvideo.io
from argparse import ArgumentParser
import time
import torch
import yaml
from tqdm.notebook import tqdm
import skimage
import skimage.filters
import skimage.io
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float, img_as_ubyte
from skimage.color import rgb2hsv, hsv2rgb
from skimage.filters import unsharp_mask
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import itertools
import skimage.metrics
from skimage.metrics import mean_squared_error
import shlex
import subprocess
import pandas as pd
from multiprocessing import Pool
#from line_profiler import LineProfiler
import cv2
import shutil
import random
import compressai
import math
from tools.early_stopping import EarlyStopping
from tools.save_model import SaveBestHandler
from compressai.zoo import bmshj2018_factorized, cheng2020_attn, mbt2018#,ssf2020
import torchvision.transforms
import numpy as np
from IPython.display import clear_output
from piq import PieAPP
import torch.nn as nn
from torch import nn
from torchvision.transforms.functional import resize, to_tensor, normalize
#from CNNfeatures import get_features
#from VQAmodel import VQAModel
import torch.optim as optim
import os
from fr_models import AHIQ, CKDN, CKDNr, STLPIPS, WADIQAM, IQT
from fr_models import VTAMIQ, twoStepQA, CONITRIQUE,WResNet, MRperceptual
from fr_models import IQATransformerBNS, IQAConformerBNS 

try:
    cfg
except Exception:
    cfg = {"general" : {"device" : "cuda:0",  'project_dir' : "."}, "run" : {"device" : "cuda:0"}, "WRONG" : "WRONG"}
    cfg["run"]["device"] = "cuda:0"
    cfg["general"]["device"] = "cuda:0"
    device = "cuda:0"
    cfg['general']['project_dir']

if not "device" in cfg["run"]:
    device = "cuda:0"

sys.path.insert(1, '..')


def Identity(img,arg1,arg2):
    return img

def axis_swaper(func):
    def wrapper(*args):
        args = np.swapaxes(args[0],0 , -1), *args[1:]
        return np.swapaxes(func(*args),0, -1)
    return wrapper

try:
    cfg['general']['patch_sz']
except Exception:
    cfg['general']['patch_sz'] = 256

try:
    dst_dir
except Exception:
    dst_dir = "P:/7videos/"  

class enhance_Identity(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def named_parameters(self):
        return {("3.quantiles",torch.nn.Parameter(torch.tensor([[0.]]))) : torch.nn.Parameter(torch.tensor([[0.]]))} 
    def parameters(self):
        return [torch.nn.Parameter(torch.tensor([[0.]]))]
    def forward(self, X):
        return X
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self


def calculate_met(times = 1, cfg = cfg):
    for to_train in [True, False]:
        tqdm_dataset = tqdm(dataset_train if to_train else dataset_test)
        for frame in tqdm_dataset:
            if not cfg["general"]["optimize_image"]:
                X = frame.to(cfg["run"]["device"])
                Y = X.clone().detach().to(cfg["run"]["device"])
            X_enhance = enhance_Identity(X)
            X_enhance.data.clamp_(min=0,max=1)
            X_out = cfg["run"]["net_codec"].forward(X_enhance)
            X_out['x_hat'].data.clamp_(min=0,max=1)
            loss = cfg["run"]["loss_calc"](X_out, Y)
            for j in list(loss.keys()):
                j_converted = j + ("_test" if not to_train else "")
                if not j_converted in logs_plot_cur:
                    logs_plot_cur[j_converted] = []
                logs_plot_cur[j_converted].append(loss[j].data.to("cpu").numpy())
        for t in range(times):
            for j in list(logs_plot_cur.keys()):
                if not j in logs_plot:
                    logs_plot[j] = []
                logs_plot[j].append(np.mean(logs_plot_cur[j]))
        return logs_plot

class codec_Blur(nn.Module):
    def __init__(self, cfg, kernel_sizes = (3, 5) , sigma = (0.1, 2.0)):
        import torchvision
        super().__init__()
        import pickle
        self.cfg = {"general": cfg['general']}
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma 
        if self.kernel_sizes[0] == self.kernel_sizes[1]:
            self.convert_f = torchvision.transforms.GaussianBlur(kernel_size = self.kernel_sizes[0], sigma = self.sigma)
        self.X_hat = None
        #self.tmp = nn.Sequential(nn.ReLU(inplace=True),)
        with open(os.path.join(self.cfg["general"]["project_dir"], 'sample_data/likelihoods.pkl'), 'rb') as f:
            self.X_hat = pickle.load(f)
        self.X_out = {"likelihoods": self.X_hat}
        class entropy_bottleneck:
            def __init__(self):
                self.loss = lambda : 0
        self.entropy_bottleneck = entropy_bottleneck()
        self.entropy_bottleneck.loss = lambda : 0
    def named_parameters(self):
        return {("3.quantiles",torch.nn.Parameter(torch.tensor([[0.]]))) : torch.nn.Parameter(torch.tensor([[0.]]))} 
    
    def forward(self, X):
        if self.kernel_sizes[0] != self.kernel_sizes[1]:
            self.convert_f = torchvision.transforms.GaussianBlur(kernel_size = np.random.randint(self.kernel_sizes[0], self.kernel_sizes[1]) // 2 * 2 + 1, sigma = self.sigma)
        self.X_out['x_hat'] = self.convert_f(X)
        return self.X_out
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self



enhance_Identity = enhance_Identity()
class codec_Identity(nn.Module):
    def __init__(self, cfg = cfg):
        super().__init__()
        import pickle
        self.X_hat = None
        #self.tmp = nn.Sequential(nn.ReLU(inplace=True),)
        with open(os.path.join(cfg["general"]["project_dir"], 'sample_data/likelihoods.pkl'), 'rb') as f:
            self.X_hat = pickle.load(f)
        self.X_out = {"likelihoods": self.X_hat}
        class entropy_bottleneck:
            def __init__(self):
                self.loss = lambda : 0
        self.entropy_bottleneck = entropy_bottleneck()
        self.entropy_bottleneck.loss = lambda : 0
    def named_parameters(self):
        return {("3.quantiles",torch.nn.Parameter(torch.tensor([[0.]]))) : torch.nn.Parameter(torch.tensor([[0.]]))}
    def forward(self, X):
        self.X_out['x_hat'] = X
        return self.X_out
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self
codec_Identity = codec_Identity()    
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

'''
class koniq(nn.Module):# TODO: FIX  inference
    def __init__(self, model_dir ="./koniq/", device = cfg["run"]["device"], to_train = True, to_crop = False):
        super().__init__()
        import sys
        sys.path.insert(1, model_dir)
        from inceptionresnetv2 import inceptionresnetv2
        self.to_crop = to_crop
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
            
        class model_qa(nn.Module):
            def __init__(self, num_classes, **kwargs):
                super(model_qa,self).__init__()
                base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
                self.base= nn.Sequential(*list(base_model.children())[:-1])
                self.fc = nn.Sequential(
                    nn.Linear(1536, 2048),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(2048),
                    nn.Dropout(p=0.25),
                    nn.Linear(2048, 1024),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(p=0.25),
                    nn.Linear(1024, 256),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(256),         
                    nn.Dropout(p=0.5),
                    nn.Linear(256, num_classes),
                )
        
            def forward(self,x):
                x = self.base(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x    
        
        self.KonCept512 = model_qa(num_classes=1) 
        self.KonCept512.load_state_dict(torch.load(model_dir + 'KonCept512.pth'))
        self.KonCept512 = self.KonCept512.eval().to(device)
        if to_train:
            for i in self.parameters():
                i.requires_grad_(True)
    def forward(self, im, device = cfg["run"]["device"]):
        """patch size must be >= (299,299), batch_sz >= 2"""
        with self.train_eval_mode():
            if self.to_crop:
                out = self.KonCept512(im[...,:512,:512]).mean() / 100.
            else:
                out = self.KonCept512(im).mean() / 100.
        return out
import torch.nn as nn
from torchvision.transforms.functional import resize, to_tensor, normalize

class Linearity(nn.Module):
    def __init__(self, model_dir = "./LinearityIQA/LinearityIQA/", device = cfg["run"]["device"], to_train = True):
        super().__init__()
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        with self.train_eval_mode():
            sys.path.insert(1, model_dir)
            from IQAmodel import IQAModel
            self.model = IQAModel(device = device).to(device)
            checkpoint = torch.load(model_dir +"../p1q2.pth")
            self.k = checkpoint['k']
            self.b = checkpoint['b']
            self.model.load_state_dict(checkpoint['model'])
            self.model = self.model.to(device)
            del checkpoint
        if to_train:
            self.requires_grad_(True)
        self.eval()
    def forward(self, im, device = cfg["run"]["device"]):
        im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with self.train_eval_mode():
            y = self.model(im)
            val = (y[-1]* self.k[-1] + self.b[-1]).mean()
        return val / 100.
    
def Linearity_met(im, device = cfg["run"]["device"],  model_dir = "./LinearityIQA/LinearityIQA/", to_train = True):
    sys.path.insert(1, model_dir)
    from IQAmodel import IQAModel
    model = IQAModel().to(device) 
    im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    checkpoint = torch.load("./LinearityIQA/LinearityIQA/../p1q2.pth")
    model.load_state_dict(checkpoint['model'])
    y = model(im)
    #y = model(im.unsqueeze(0))
    k = checkpoint['k']
    b = checkpoint['b']
    return -(y[-1] * k[-1] + b[-1]).mean() / 100.

class NIMA(nn.Module):
    def __init__(self, model_dir ="./Neural-IMage-Assessment/", device = cfg["run"]["device"], crop = False, to_train = True):
        super().__init__()
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        import sys
        import torchvision.models as models
        import torchvision.transforms as transforms
        sys.path.insert(1, model_dir)
        from model.model import NIMA   
        self.crop = crop
        self.base_model = models.vgg16(pretrained=True).to(device)
        self.model = NIMA(self.base_model).to(device)
        self.model.load_state_dict(torch.load(model_dir + "model/epoch-82.pth"))
        self.eval()
        for i in self.parameters():
            i.requires_grad_(True)
    def forward(self, im, device = cfg["run"]["device"]):
        transforms = torchvision.transforms
        with self.train_eval_mode():
            if self.crop:
                out = self.model(im[:,:,:224,:224])
            else:
                val_transform = transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])])
                im = val_transform(im)
                out = self.model(im)
        out = out.view(-1, 10, 1)
        mean = 0
        batch_sz_0 = len(out)
        for out_i in out:
            for j, e in enumerate(out_i, 1):
                mean += j * e
        return mean / batch_sz_0 / 10.
    

class VSFA_loss(nn.Module):#TODO: check [0][0]
    def __init__(self, model_dir = "./VSFA/VSFA/", device = cfg["run"]["device"], to_train = True):
        super().__init__()
        import sys
        sys.path.insert(1, model_dir)
        import VSFA
        from CNNfeatures import get_features
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        self.get_features = get_features
        self.model = VSFA.VSFA()
        self.model.load_state_dict(torch.load(model_dir + "models/VSFA.pt"))
        self.model.eval()
        self.model.to(device)
    def forward(self, X_sample, device = cfg["run"]["device"]):
        with self.train_eval_mode():
            self.features = self.get_features(X_sample, frame_batch_size = len(X_sample), device=device)
            self.features = torch.unsqueeze(self.features, 0)  # batch size 1
            input_length = self.features.shape[1] * torch.ones(1, 1)
            outputs = self.model(self.features, input_length)
        return outputs[0][0]

class BRISQ(nn.Module):
    def __init__(self, device = cfg["run"]["device"], to_train = True):
        super().__init__()
        from piq import BRISQUELoss
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        self.model = BRISQUELoss()
        self.model.eval()
        self.model = self.model.to(device)
    def forward(self, X_sample):
        with self.train_eval_mode():
            val = self.model(torch.clamp(X_sample,0,1))
        return val

class SPAQ(nn.Module):#OK
    def __init__(self, model_dir = "./SPAQ", device = cfg["run"]["device"], to_train = True):
        super().__init__()
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        sys.path.insert(1, model_dir)
        from BL_demo import Demo#Changed map_location
        self.dm = Demo("", checkpoint_dir='./SPAQ/weights/BL_release.pt', device = device )
        self.dm.model = self.dm.model.to(device)
    def forward(self, im, device = cfg["run"]["device"]):
        with self.train_eval_mode():
            score_1 = self.dm.model(im).mean()
        return score_1 / 100.

class paq2piq_model(nn.Module):#OK
    def __init__(self, model_dir = "./paq2piq/", device = cfg["run"]["device"], blk_size = None, to_train = True):
        super().__init__()
        import sys
        sys.path.insert(1,model_dir)
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        from paq2piq_standalone import InferenceModel, RoIPoolModel
        self.model = InferenceModel(RoIPoolModel(backbone='resnet18', pretrained=True), model_dir + "models/RoIPoolModel-fit.10.bs.120.pth", device = device)
        if blk_size != None:
            self.model.blk_size = blk_size
        
    def forward(self, X_sample, device = cfg["run"]["device"]):
        with self.train_eval_mode():
            batch_sz = len(X_sample)
            global_score_batch = 0
            for X_i in X_sample:
                t = self.model.model(X_i.unsqueeze(0))[0]
                self.model.model.input_block_rois(self.model.blk_size, [X_sample.shape[-2], X_sample.shape[-1]], device=device)
                global_score = t[0]
                global_score_batch += global_score
            global_score_batch = global_score_batch / batch_sz /100.
        return global_score_batch
'''

class Logger():
    def __init__(self, cfg):
        self.cfg = {"general" : cfg["general"]}
        try:
            os.mkdir(os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"]))
        except Exception:
            pass
    def write_cfg(self, ):
        with open(os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "cfg.yaml"), "w") as fh:  
            yaml.dump({"general":self.cfg["general"]}, fh)
    def write_code(self, ):
        try:
            os.mkdir(os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "code"))
        except Exception:
            pass
        src = os.path.join(self.cfg["general"]["project_dir"],'Current_model_lib.py')
        dst = os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "code",'Current_model_lib.py')
        shutil.copyfile(src, dst)
        src = os.path.join(self.cfg["general"]["project_dir"],'Train_current_model.py')
        dst = os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "code",'Train_current_model.py')
        shutil.copyfile(src, dst)
        #src = os.path.join(self.cfg["general"]["project_dir"],'main.py')
        #dst = os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "code", 'main.py')
        #shutil.copyfile(src, dst)
    def write_logs(self, ):
        try:
            with open(os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "logs.yaml"), "w") as fh:  
                yaml.dump({"logs" : self.cfg["logs"]}, fh)
        except Exception:
            print("exception while logging yaml")
        try:
            np.save(os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "logs.npy"), Log_1)
        except Exception:
            print("exception while logging npy")
    def save_img(self, args):
        pltimshow_batch(args, filename = os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"], "vis.png"))
    def save_img_from_dataset(self, X, img_nam, subdir = None):
        imgs_path = os.path.join(self.cfg["general"]["logs_dir"], self.cfg["general"]["name"],"imgs")
        
        if not os.path.exists(imgs_path):
            os.mkdir(imgs_path)
        if subdir == None:
            imgs_path_subdir = imgs_path
        else:
            imgs_path_subdir = os.path.join(imgs_path, subdir)
        if not os.path.exists(imgs_path_subdir):
            os.mkdir(imgs_path_subdir)
        img_full_path = os.path.join(imgs_path_subdir, img_nam)
        torchvision.utils.save_image(X, img_full_path)  


from piq import LPIPS as piq_LPIPS#PieAPP VSI, FSIM, NLPD, deepIQA
from piq import DISTS as piq_DISTS
import IQA_pytorch as iqa#SSIM, GMSD, LPIPSvgg, DISTS
class calc_met:
    def __init__(self,dataset1 = ["Run439.Y4M"], convKer1 = None, home_dir1 = "R:/", creat_dir = False, calc_SSIM_PSNR = False, calc_model_features = False,device = cfg["run"]["device"], model = "vmaf_v063" , codec = '   -preset:v medium -x265-params log-level=error ',dataset_dir = "dataset/", to_train = True):
        self.device = device
        if to_train:
            self.train_eval_mode = torch.enable_grad
        else:
            self.train_eval_mode = torch.no_grad
        self.model = VQAModel().to(device)
        self.model.load_state_dict(torch.load('../models/MDTVSFA.pt'))
        self.model.eval()
        self.frame_batch_size = 1
        self.dataset_err = None
        self.dataset_err_torch = None
        self.dataset_np = None
        self.dataset_torch = None
        self.datagen = None
        self.features = None
        self.dataset = []
        self.crf_arr = []
        self.dataset_dir = dataset_dir
        self.calc_model_features = calc_model_features
        self.Results = []
        self.relative_score, self.mapped_score, self.aligned_score = 0,0,0
    def MDTVSFA(self, transformed_video):
        with self.train_eval_mode():
            self.features = get_features(transformed_video, frame_batch_size=self.frame_batch_size, device=self.device)
            self.features = torch.unsqueeze(self.features, 0) 
            if len(self.features.shape) == 2:
                self.features = self.features.unsqueeze(0)
            input_length = self.features.shape[1] * torch.ones(1, 1, dtype=torch.long)
            self.relative_score, self.mapped_score, self.aligned_score = self.model.forward([(self.features, input_length, ['K'])])
            y_pred = self.mapped_score[0][0]#.to('cpu').detach().numpy()
        return y_pred


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["mse"] = self.mse(output["x_hat"], target)
        try:
            out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
            )
            out["loss_classic"] = self.lmbda * 255 ** 2 * out["mse"] + out["bpp_loss"]
        except Exception:
            pass
        return out
# DSS, SRSIM
class Custom_enh_Loss(nn.Module):
    def __init__(self, lmbda=1e-2, device = cfg["run"]["device"], target_lst = ["VSFA", "mse"], k_lst = [1, 2000.], to_train = True, crop_NIMA = True, to_crop_koniq = False):
        super().__init__()
        self.k_lst = k_lst
        self.target_lst = target_lst
        self.rdLoss = RateDistortionLoss(lmbda)
        if "LPIPS" in self.target_lst:
            self.lpips = iqa.LPIPSvgg().to(device)
        if "SSIM" in self.target_lst:
            self.ssim = iqa.SSIM()
        if "DISTS" in self.target_lst:
            self.dists = iqa.DISTS().to(device)
        if "MDTVSFA" in self.target_lst:
            self.MDTVSFA_metr = calc_met(to_train = to_train)
        if "BRISQ" in self.target_lst:
            self.brisq_loss = BRISQ(to_train = to_train)   
        if "Linearity_tmp" in self.target_lst:
            self.lin_loss = Linearity_met
        if "Linearity" in self.target_lst:
            self.lin_loss = Linearity(to_train = to_train)
            self.lin_loss = self.lin_loss.to(device)
        if "SPAQ" in self.target_lst:
            self.spaq_loss = SPAQ(to_train = to_train)
        if "VSFA" in self.target_lst:
            self.vsfa_loss = VSFA_loss(to_train = to_train)
        if "PieAPP" in self.target_lst:
            self.piapp_loss = PieAPP(enable_grad = to_train)
        if "PAC2PIQ" in self.target_lst:
            self.paq2piq_loss = paq2piq_model(to_train = to_train)
        if "NIMA" in self.target_lst:
            self.NIMA_loss = NIMA(to_train = to_train, crop = crop_NIMA)
        if "KONIQ" in self.target_lst:
            self.koniq_loss = koniq(to_train = to_train, to_crop = to_crop_koniq)
        #Full-ref
        from piq import HaarPSILoss,VIFLoss, DSSLoss,dss, multi_scale_ssim, multi_scale_gmsd,vif, vif_p,MDSILoss, GMSDLoss,VSILoss,SRSIMLoss, InformationWeightedSSIMLoss
        from IQA_pytorch import GMSD, VIF, VIFs, MS_SSIM
        from piq import DISTS as piq_DISTS
        from piq import LPIPS as piq_LPIPS
        import IQA_pytorch
        import piq
        if "GMSD" in self.target_lst:
            self.GMSD_loss = GMSDLoss()
        if "GMSD1" in self.target_lst:
            self.GMSD_loss_1 = GMSD()
        if "VIFs" in self.target_lst:
            self.iqa_vifs_loss = VIFs()
        if "VIF" in self.target_lst:
            self.iqa_vif_loss = VIF()
        if "VIFLoss" in self.target_lst:
            self.piq_vif_Loss = VIFLoss()
        if "VIFp" in self.target_lst:
            self.piq_vif_p_loss = vif_p
        if "DSS" in self.target_lst: 
            self.dss_loss_1 = DSSLoss()
        if "DSS1" in self.target_lst: 
            self.dss_loss_2 = dss
        if "MS-SSIM1" in self.target_lst: 
            self.msssim_loss_1 = multi_scale_ssim
        if "HaarPSI" in self.target_lst: 
            self.haarpsi_loss = HaarPSILoss()
        if "MS-SSIM" in self.target_lst:
            self.msssim_loss_2 = MS_SSIM()
        if "MS-GMSD" in self.target_lst:
            self.ms_gmsd_loss = MultiScaleGMSDLoss()#multi_scale_gmsd
        if "SRSIM" in self.target_lst:
            self.SRSIM_loss = SRSIMLoss()
        if "VSI" in self.target_lst:
            self.VSI_loss = VSILoss()
        if "MDSI" in self.target_lst:
            self.MDSI_met = MDSILoss()
        if "DISTS1" in self.target_lst:
            self.DISTS_loss_2 = piq_DISTS()
        if "LPIPS1" in self.target_lst:
            self.LPIPS_loss_2 = piq_LPIPS()
        if "FSIM" in self.target_lst:
            self.FSIMLoss = piq.FSIMLoss()
        if "StyleLoss" in self.target_lst:
            self.StyleLoss = piq.StyleLoss()
        #if "SF" in self.target_lst:
            #self.sf = piq.sf()
        if "NLPD" in self.target_lst:
            self.NLPD = IQA_pytorch.NLPD()
        if "ContentLoss" in self.target_lst:
            self.ContentLoss = piq.ContentLoss()

        if "AHIQ" in self.target_lst:
            self.AHIQ = AHIQ()
        if "CKDN" in self.target_lst:
            self.CKDN = CKDN()
        if "CKDNr" in self.target_lst:
            self.CKDNr = CKDNr()
        if "STLPIPS" in self.target_lst:
            self.STLPIPS = STLPIPS()
        if "WADIQAM" in self.target_lst:
            self.WADIQAM = WADIQAM()
        if "IQT" in self.target_lst:
            self.IQT = IQT()
        if "VTAMIQ" in self.target_lst:
            self.VTAMIQ = VTAMIQ()
        if "twoStepQA" in self.target_lst:
            self.twoStepQA = twoStepQA()
        if "CONITRIQUE" in self.target_lst: 
            self.CONITRIQUE = CONITRIQUE()
        if "WResNet" in self.target_lst:
            self.WResNet = WResNet()
        if "MRperceptual" in self.target_lst:
            self.MRperceptual = MRperceptual()
        if "IQAConformerBNS" in self.target_lst:
            self.IQAConformerBNS = IQAConformerBNS()
        if "IQATransformerBNS" in self.target_lst:
            self.IQATransformerBNS = IQATransformerBNS()
        if "DISTScrops1" in self.target_lst:
            from fr_models import DISTScrops
            self.dists_crops1 = DISTScrops(device = "cuda:0", mode = 'mrpl', crop_sz = 192, crops_num = 1).to(device)
        if "DISTScrops2" in self.target_lst:
            from fr_models import DISTScrops
            self.dists_crops2 = DISTScrops(device = "cuda:0", mode = 'mrpl', crop_sz = 32, crops_num = 64).to(device)
        if "IW-SSIM" in self.target_lst:
            self.iwssim = InformationWeightedSSIMLoss()   
        if "CW-SSIM" in self.target_lst:
            #from pyiqa.archs import CW_SSIM
            from pyiqa.archs.ssim_arch import CW_SSIM
            self.cw_ssim = CW_SSIM()
        if "MAD" in self.target_lst:
            from pyiqa.archs.mad_arch import MAD
            self.mad_loss = MAD()
        if "ADISTS" in self.target_lst:
            from ADISTS.ADISTS_pt import ADISTS
            self.adists = ADISTS().to(device)
        if "MACS" in self.target_lst:
            #from MACS.siamunet_diff import SiamUnet_diff
            from MACS.siamunet_conc import SiamUnet_conc
            self.macs = SiamUnet_conc(3, 1)

            self.macs.load_state_dict(torch.load("MACS/ch_net-best_epoch-129.pth.tar")['model_state_dict'])
            self.macs = self.macs.to(device).eval()
            

    def forward(self, X_out, Y):
        if X_out['x_hat'].device != Y.device:
            X_out['x_hat'] = X_out['x_hat'].to(Y.device)
        self.loss = self.rdLoss(X_out, Y)
        if 'bpp_loss' in X_out.keys():
            self.loss['bpp_loss'] = X_out['bpp_loss']

        self.loss['PSNR'] = 10 * torch.log10(1. / self.loss['mse'])
        if "LPIPS" in self.target_lst:
            self.loss["LPIPS"] = self.lpips(X_out['x_hat'], Y)
        if "SSIM" in self.target_lst:
            self.loss["SSIM"] = self.ssim(Y, X_out['x_hat'])
        if "DISTS" in self.target_lst:
            self.loss["DISTS"] = self.dists(X_out['x_hat'], Y)
        if "MDTVSFA" in self.target_lst:
            self.loss['MDTVSFA'] = -self.MDTVSFA_metr.MDTVSFA(X_out['x_hat']).mean()
        if "BRISQ" in self.target_lst:
            self.loss["BRISQ"] = self.brisq_loss(X_out['x_hat'])
        if "Linearity" in self.target_lst:
            self.loss["Linearity"] = -self.lin_loss(X_out['x_hat'])
        if "Linearity_tmp" in self.target_lst:
            self.loss['Linearity_tmp'] = self.lin_loss(X_out['x_hat'])
        if "SPAQ" in self.target_lst:
            self.loss["SPAQ"] = -self.spaq_loss(X_out['x_hat'])
        if "VSFA" in self.target_lst:
            self.loss["VSFA"] = -self.vsfa_loss(X_out['x_hat'])
        if "PieAPP" in self.target_lst:
            self.loss["PieAPP"] = self.piapp_loss(X_out['x_hat'], Y).mean()
        if "PAC2PIQ" in self.target_lst:
            self.loss["PAC2PIQ"] = -self.paq2piq_loss(X_out['x_hat'])
        if "NIMA" in self.target_lst:
            self.loss["NIMA"] = -self.NIMA_loss(X_out['x_hat']).mean()
        if "KONIQ" in self.target_lst:
            self.loss["KONIQ"] = -self.koniq_loss(X_out['x_hat'])
        #Full-ref    
        if "GMSD" in self.target_lst:
            self.GMSD_loss.train()
            X1 = torch.minimum(torch.maximum(X_out['x_hat'], torch.zeros_like(X_out['x_hat'], device = X_out['x_hat'].device)), torch.ones_like(X_out['x_hat'], device = X_out['x_hat'].device))
            Y1 = torch.minimum(torch.maximum(Y, torch.zeros_like(Y, device = Y.device)), torch.ones_like(Y, device = Y.device))
            self.loss["GMSD"] = self.GMSD_loss(X1, Y1)
        if "GMSD1" in self.target_lst:
            self.loss["GMSD1"] = self.GMSD_loss_1(X_out['x_hat'], Y)
        if "VIFs" in self.target_lst:
            self.loss["VIFs"] = self.iqa_vifs_loss(X_out['x_hat'], Y)
        if "VIF" in self.target_lst:
            self.loss["VIF"] = self.iqa_vif_loss(X_out['x_hat'], Y)
        if "VIFLoss" in self.target_lst:
            self.loss["VIFLoss"] = self.piq_vif_Loss(X_out['x_hat'], Y)
        if "VIFp" in self.target_lst:
            self.loss["VIFp"] = 1 - self.piq_vif_p_loss(X_out['x_hat'], Y)#higher -- better
        if "DSS" in self.target_lst: 
            X1 = torch.minimum(torch.maximum(X_out['x_hat'], torch.zeros_like(X_out['x_hat'], device = X_out['x_hat'].device)), torch.ones_like(X_out['x_hat'], device = X_out['x_hat'].device))
            Y1 = torch.minimum(torch.maximum(Y, torch.zeros_like(Y, device = Y.device)), torch.ones_like(Y, device = Y.device))
            self.loss["DSS"] = self.dss_loss_1(X1,Y1)
        if "DSS1" in self.target_lst: 
            self.loss["DSS1"] =  1 - self.dss_loss_2(X_out['x_hat'], Y)#higher -- better
        if "MS-SSIM1" in self.target_lst: 
            self.loss["MS-SSIM1"] = self.msssim_loss_1(X_out['x_hat'], Y)
        if "HaarPSI" in self.target_lst: 
            self.loss["HaarPSI"] = self.haarpsi_loss(X_out['x_hat'], Y)
        if "MS-SSIM" in self.target_lst:
            self.loss["MS-SSIM"] = self.msssim_loss_2(X_out['x_hat'], Y)
        if "MS-GMSD" in self.target_lst:
            self.loss["MS-GMSD"] = self.ms_gmsd_loss(X_out['x_hat'], Y)
        if "SRSIM" in self.target_lst:
            self.loss["SRSIM"] = self.SRSIM_loss(X_out['x_hat'], Y)
        if "VSI" in self.target_lst:
            self.loss["VSI"] = self.VSI_loss(X_out['x_hat'], Y)
        if "MDSI" in self.target_lst:
            self.loss["MDSI"] =  1 - self.MDSI_met(X_out['x_hat'], Y)#higher -- better
        if "DISTS1" in self.target_lst:
            self.loss["DISTS1"] = self.DISTS_loss_2(X_out['x_hat'], Y)
        if "LPIPS1" in self.target_lst:
            self.loss["LPIPS1"] = self.LPIPS_loss_2(X_out['x_hat'], Y)
        if "FSIM" in self.target_lst:
            self.loss["FSIM"] = self.FSIMLoss(X_out['x_hat'], Y)
        if "StyleLoss" in self.target_lst:
            self.loss["StyleLoss"] = self.StyleLoss(X_out['x_hat'], Y)
        #if "SF" in self.target_lst:
           # self.loss["SF"] = self.sf(X_out['x_hat'], Y)   
        if "NLPD" in self.target_lst:
            self.loss["NLPD"] = self.NLPD(X_out['x_hat'], Y)    
        if "ContentLoss" in self.target_lst:
            self.loss["ContentLoss"] = self.ContentLoss(X_out['x_hat'], Y) 
    
        if "AHIQ" in self.target_lst:
            self.loss["AHIQ"] = self.AHIQ(X_out['x_hat'], Y)
        if "CKDN" in self.target_lst:
            self.loss["CKDN"] = self.CKDN(X_out['x_hat'], Y)
        if "CKDNr" in self.target_lst:
            self.loss["CKDNr"] = self.CKDNr(X_out['x_hat'], Y)
        if "STLPIPS" in self.target_lst:
            self.loss["STLPIPS"] = self.STLPIPS(X_out['x_hat'], Y)
        if "WADIQAM" in self.target_lst:
            self.loss["WADIQAM"] = self.WADIQAM(X_out['x_hat'], Y)
        if "IQT" in self.target_lst:
            self.loss["IQT"] = self.IQT(X_out['x_hat'], Y)
        if "VTAMIQ" in self.target_lst:
            self.loss["VTAMIQ"] = self.VTAMIQ(X_out['x_hat'], Y)
        if "twoStepQA" in self.target_lst:
            self.loss["twoStepQA"] = self.twoStepQA(X_out['x_hat'], Y)
        if "CONITRIQUE" in self.target_lst:
            self.loss["CONITRIQUE"] = self.CONITRIQUE(X_out['x_hat'], Y)
        if "WResNet" in self.target_lst:
            self.loss["WResNet"] = self.WResNet(X_out['x_hat'], Y)
        if "MRperceptual" in self.target_lst:
            self.loss["MRperceptual"] = self.MRperceptual(X_out['x_hat'], Y)
        if "IQAConformerBNS" in self.target_lst:
            self.loss["IQAConformerBNS"] = self.IQAConformerBNS(X_out['x_hat'], Y)
        if "IQATransformerBNS" in self.target_lst:
            self.loss["IQATransformerBNS"] = self.IQATransformerBNS(X_out['x_hat'], Y)
        if "DISTScrops1" in self.target_lst:
            self.loss["DISTScrops1"] = self.dists_crops1(X_out['x_hat'], Y)
        if "DISTScrops2" in self.target_lst:
            self.loss["DISTScrops2"] = self.dists_crops2(X_out['x_hat'], Y)  
        if "IW-SSIM" in self.target_lst:
            Yc = torch.minimum(torch.maximum(Y, torch.zeros_like(Y, device = Y.device)), torch.ones_like(Y, device = Y.device))
            Xc = torch.minimum(torch.maximum(X_out['x_hat'], torch.zeros_like(X_out['x_hat'], device = X_out['x_hat'].device)), torch.ones_like(X_out['x_hat'], device = X_out['x_hat'].device))
            self.loss["IW-SSIM"] = self.iwssim(Xc, Yc)
        if "CW-SSIM" in self.target_lst:
            self.loss["CW-SSIM"] = torch.mean(1-self.cw_ssim(X_out['x_hat'], Y))
        if "MAD" in self.target_lst:
            self.loss["MAD"] = self.mad_loss(X_out['x_hat'], Y)
        if "ADISTS" in self.target_lst:
            self.loss["ADISTS"] = torch.mean(self.adists(X_out['x_hat'], Y))
        if "MACS" in self.target_lst:#not working
            self.loss["MACS"] = torch.mean(self.macs(X_out['x_hat'], Y))
        self.loss["loss"] = 0
        for cur_metrics, k in zip(self.target_lst, self.k_lst):
            self.loss["loss"] += k * self.loss[cur_metrics]
        return self.loss

class Custom_enh_Loss_seq(nn.Module):
    def __init__(self, lmbda=1e-2, device = cfg["run"]["device"], target_lst = ["VSFA", "mse"], k_lst = [1, 2000.]):
        super().__init__()
        self.k_lst = k_lst
        self.target_lst = target_lst
        self.rdLoss = RateDistortionLoss(lmbda)
    def forward(self, X_out, Y):
        global_loss = {}
        for idx in range(len(self.target_lst)):
            cur_loss_f = Custom_enh_Loss(target_lst = [self.target_lst[idx]], device = device, k_lst = [self.k_lst[idx]]).to(device)
            cur_loss = cur_loss_f(X_out, Y)
            global_loss.update(cur_loss)
            del cur_loss_f
        return global_loss



class Video_reader_read():
    def __init__(self,name1):
        self.nameGT = name1
        
    def get_frame(self):
        self.temp_reader1 = skvideo.io.FFmpegReader(self.nameGT, outputdict={"-c:v" :" rawvideo","-f": "rawvideo"})
        self.datagenGT = [frameGT / 255. for frameGT in self.temp_reader1.nextFrame()]
        self.temp_reader1.close()
        self.datagenGT = np.array([[i[:,:,0],i[:,:,1],i[:,:,2]] for i in self.datagenGT])
        self.lst_1 = torch.tensor(self.datagenGT[0]).float() - 0.5
        return torch.stack([self.lst_1])
    
    def get_frames(self):
        self.temp_reader1 = skvideo.io.FFmpegReader(self.nameGT, outputdict={"-c:v" :" rawvideo","-f": "rawvideo"})
        self.datagenGT = [frameGT / 255. for frameGT in self.temp_reader1.nextFrame()]
        self.temp_reader1.close()
        self.datagenGT = np.array([[i[:,:,0],i[:,:,1],i[:,:,2]] for i in self.datagenGT])
        self.lst_1 = torch.tensor(self.datagenGT).float() - 0.5
        return self.lst_1
    
def pltimshow(arg):
    plt.imshow(arg.cpu().detach().numpy().swapaxes(1,3).swapaxes(1,2)[0])

def pltimshow_batch(args, filename = "vis/tmp.png"):
    plt.figure(dpi = 800)
    Ar2 = None
    for arg in args:
        Ar1 = np.concatenate([i for i in arg.cpu().detach().numpy().swapaxes(1,3).swapaxes(1,2)], 0)
        if Ar2 is None:
            Ar2 = Ar1
        else:
            Ar2 = np.concatenate([Ar2,Ar1],1)
    
    plt.imshow(Ar2)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')

rdLoss = RateDistortionLoss()
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torchvision
def dir_of_dirs(paths):
    A = []
    for j in paths:
        for i in os.listdir(j):
            A.append(os.path.join(j, i))
    return A


class Video_reader_dataset(Dataset):
    def __init__(self, name1, num_frames = None, minimal_batch_sz = 0):
        super(CustomImageDataset).__init__()
        self.nameGT = name1
        self.temp_reader1 = skvideo.io.FFmpegReader(self.nameGT, outputdict={"-c:v" :" rawvideo","-f": "rawvideo"})
        self.datalen = self.temp_reader1.getShape()[0]
        #self.datagenGT = [frameGT / 255. for frameGT in self.temp_reader1.nextFrame()]
        self.datagenGT = self.temp_reader1.nextFrame()# [frameGT / 255. for frameGT in ]
        #self.temp_reader1.close()
        
        if num_frames != None:
            self.datalen = min(self.datalen, num_frames)
        if minimal_batch_sz:
            self.datalen = self.datalen // minimal_batch_sz * minimal_batch_sz
    def __len__(self):
        return self.datalen
    def close(self):
        self.temp_reader1.close()
    def __getitem__(self, idx):
        if idx >= self.datalen:
            self.temp_reader1.close()
            raise StopIteration
        self.frame = next(self.datagenGT) / 255. #self.lst_1[idx]
        self.frame = np.array([self.frame[:,:,0], self.frame[:,:,1], self.frame[:,:,2]])
        self.frame = torch.tensor(self.frame).float() 
        return self.frame 

    
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, cfg, transform=None, target_transform=None,train = True, datalen = 128, center_crop = False, return_name = False):
        super(CustomImageDataset).__init__()
        self.cfg = {"general" : cfg["general"]}
        self.return_name = return_name
        self.center_crop = center_crop
        self.train = train
        self.image = 0
        self.label = 0
        self.img_names = dir_of_dirs(dir_of_dirs(dir_of_dirs([img_dir])))
        self.datalen = min(datalen, len(self.img_names))
        self.img_dir = img_dir
    def __len__(self):
        return self.datalen#9600#len(self.img_names)
    def __getitem__(self, idx):
        if not self.train:
            idx = len(self.img_names) - idx - 1
        img_path = self.img_names[idx]
        image = read_image(img_path)
        if len(image.shape) == 2 or image.shape[0] == 1:
            image = torch.cat([image for i in range(3)])
        self.image = image
        if self.center_crop:
            self.image = torchvision.transforms.CenterCrop((self.cfg['general']['patch_sz'], self.cfg['general']['patch_sz']))(self.image)
        else:
            self.image = torchvision.transforms.RandomResizedCrop((self.cfg['general']['patch_sz'], self.cfg['general']['patch_sz']))(self.image)
        if self.return_name:
            return self.image / 255., self.img_names[idx]
        else:
            return self.image / 255. 
    def close(self):
        del self.image
        
def max_over_iter(arg):
    max_val = 0
    from collections.abc import Iterable
    try:
        if isinstance(arg, type(np.ndarray)):
             max_val = max(max_val, max(arg_n))
        elif isinstance(arg, torch.Tensor):
            arg_n = arg.cpu().numpy()
            max_val = max(max_val, max(arg_n))
        elif isinstance(arg, dict):
            b = [j for i,j in arg.items()]
            for v in b:
                max_val = max(max_over_iter(v), max_val)
        elif isinstance(arg, Iterable):
            for i in arg:
                max_val = max(max_over_iter(i), max_val)
    except Exception:
        pass
    return max_val

def get_met(X):
    if met_name == "VSFA":
        return -cfg["run"]["loss_calc"]({"x_hat": torch.cat([X,X])}, torch.cat([X,X]))["loss"]
    else:
        return -cfg["run"]["loss_calc"]({"x_hat": X}, X)["loss"]

from piq import  MultiScaleSSIMLoss, MultiScaleGMSDLoss, GMSDLoss, FSIMLoss, VSILoss, MDSILoss, SRSIMLoss, DSSLoss, InformationWeightedSSIMLoss