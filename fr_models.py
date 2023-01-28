import torch
import torch.nn as nn
import torch.nn 


class AHIQ(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        import pyiqa
        self.model_ahiq = pyiqa.create_metric('ahiq', as_loss=True, device = device)# higher -- better, not limited
        self.model_ahiq.training = True
    def forward(self, X, Y):
        X, Y = CropPatches(X,Y,224,1)
        X,Y = X.squeeze(1), Y.squeeze(1)
        #print(X.shape)
        return -torch.mean(self.model_ahiq(X,Y))


class CKDN(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        import pyiqa
        self.model_ckdn = pyiqa.create_metric('ckdn', as_loss=True, device= device)# higher -- better, not limited
    def forward(self, X, Y):
        return -torch.mean(self.model_ckdn(Y,X))

class CKDNr(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        import pyiqa
        self.model_ckdn = pyiqa.create_metric('ckdn', as_loss=True, device= device)# higher -- better, not limited
    def forward(self, X, Y):
        return torch.mean(self.model_ckdn(X,Y))


class STLPIPS(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        from stlpips_pytorch import  stlpips
        self.model = stlpips.LPIPS(net = 'alex', variant="shift_tolerant",pretrained=True).to(device)#Lower-better https://github.com/abhijay9/ShiftTolerant-LPIPS
    def forward(self, X, Y):
        return torch.mean(self.model(X,Y))


def CropPatches(im, ref, patch_size=32, num_patches = 64):
    """
    Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    w, h = im.shape[-2:]
    batchsz = im.shape[0]
    patches_im = []
    patches_ref = []
    stride = patch_size


    h_end = h - stride
    w_end = w - stride
    import random
    
    for idx in range(num_patches):
        we, he = random.randint(0,w_end), random.randint(0,h_end)
        
        patch_im = im[:, :, we : we + patch_size, he : he + patch_size]
        patch_ref = ref[:, :, we : we + patch_size, he : he + patch_size]
        patches_im = patches_im + [patch_im, ]
        patches_ref = patches_ref + [patch_ref, ]
        
    return torch.stack(patches_im, dim = 1), torch.stack(patches_ref,dim = 1)




class WADIQAM(nn.Module):
    def __init__(self, device = "cuda:0", patches = True):
        super().__init__()
        import WaDIQaM
        self.patches = patches
        from WaDIQaM.main import RandomCropPatches, NonOverlappingCropPatches, FRnet
        self.model = FRnet(weighted_average=True).to(device)
        self.model.load_state_dict(torch.load("WaDIQaM/checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4"))
        self.model.eval()
    def forward(self, X, Y):
        if self.patches:
            patchX, patchY = CropPatches(X, Y)
            val = self.model([patchX, patchY])# needs patches
        else:
            val = self.model([X.unsqueeze(0),Y.unsqueeze(0)])
        return -torch.mean(val)

class IQT_old(nn.Module):#check patches, model load
    def __init__(self, device = "cuda:0"):
        super().__init__()# Crop 192, random, central -- easier hacking
        from IQT.test import get
        self.model = get
    def forward(self, X, Y):
        return -torch.mean(self.model(X,Y))



class IQT(nn.Module):#check patches, model load
    def __init__(self, device = "cuda:0", fixed_crop = True):
        super().__init__()# Crop 192, random, central -- easier hacking
        from IQT.test import IQTmodel
        self.model = IQTmodel(fixed_crop = fixed_crop).to(device)
    def forward(self, X, Y):
        return self.model(X,Y)




class VTAMIQ(nn.Module):#spical patches, check if diff
    def __init__(self, device = "cuda:0"):
        """
        ?used without pre-trained model weights
        """
        super().__init__()
        from VTAMIQ.run_main import get_model
        self.model = get_model(device, 'VTAMIQ/lastest.pth')
        import torch
        import torchvision.transforms
        from VTAMIQ.data.sampling.patch_sampling import get_iqa_patches, PatchSampler
        self.model.eval()
        self.sampler = PatchSampler()
        self.tensr2PIL = torchvision.transforms.ToPILImage()
    def forward(self, X, Y):
        from VTAMIQ.data.sampling.patch_sampling import get_iqa_patches
        mode = 0
        #print("BEFORE")
        #print(X.device)
        #print(Y.device)

        if mode == 0:
            patches = get_iqa_patches([self.tensr2PIL(X[0]),self.tensr2PIL(Y[0])],[X[0],Y[0]],16,(16,16),self.sampler,16)
        else:
            p1 = CropPatches(X[0].unsqueeze(0) , Y[0].unsqueeze(0), 16, 16)
            patches = get_iqa_patches([self.tensr2PIL(X[0]),self.tensr2PIL(Y[0])],[X[0],Y[0]],16,(16,16),self.sampler,16)
            patches = p1[0][0], p1[1][0], patches[1], patches[2].to(X.device), patches[3].to(X.device)
        #print("AFTER")
        #print(patches[0].device)
        #print(patches[1].device)
        #print(patches[2].device)
        #print(patches[3].device)
        val = self.model([patches[0].unsqueeze(0), patches[1].unsqueeze(0)], patches_scales = patches[3], patches_pos = patches[2])
        return -torch.mean(val)

class twoStepQA(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        import pyiqa
        import piq
        self.niqe = pyiqa.create_metric('niqe', as_loss=True)
        self.mssim = piq.ms_ssim.multi_scale_ssim
        #self.model_2stepQA = 0
        self.mssim_val = 0
        self.niqe_val = 0
        self.val_2stepQA = 0
        
    def forward(self, X, Y):
        X = torch.minimum(torch.maximum(X, torch.zeros_like(X, device = X.device)), torch.ones_like(X, device = X.device))
        Y = torch.minimum(torch.maximum(Y, torch.zeros_like(Y, device = Y.device)), torch.ones_like(Y, device = Y.device))
        self.mssim_val = self.mssim(X,Y)
        self.niqe_val = self.niqe(X,Y) 
        self.val_2stepQA = self.mssim_val * (1 - self.niqe_val / 100)
        val = self.val_2stepQA
        return -torch.mean(val)

class CONITRIQUE(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        import torch
        from CONTRIQUE.modules.network import get_network
        from CONTRIQUE.modules.CONTRIQUE_model import CONTRIQUE_model
        from torchvision import transforms
        import numpy as np
        
        import os
        import argparse
        import pickle
        from PIL import Image
        args = {"model_path" : 'CONTRIQUE/models/CONTRIQUE_checkpoint25.tar', 
        'linear_regressor_path' : 'CONTRIQUE/models/LIVE_FR.save',}
        self.encoder = get_network('resnet50', pretrained=False)
        self.model = CONTRIQUE_model(args, self.encoder, 2048)
        

        self.model.load_state_dict(torch.load(args["model_path"], map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.regressor = pickle.load(open('CONTRIQUE/models/CSIQ_FR.save', 'rb'))
        self.k = torch.tensor(self.regressor.coef_, device= device)
    def forward(self, X, Y):
        import pickle
        sz = X.shape
        from torchvision.transforms import Resize
        mode = 1
        if mode == 0:
            ref_image_2, dist_image_2 = Resize((sz[-2]//2, sz[-1] // 2))(Y), Resize((sz[-2]//2, sz[-1] // 2))(X)## CHECK if diff
        else:
            ref_image_2, dist_image_2 = Y[...,::2,::2], X[...,::2,::2]
        _,_, _, _, ref_feat, ref_feat_2, _, _ = self.model(Y, ref_image_2)
        _,_, _, _, dist_feat, dist_feat_2, _, _ = self.model(X, dist_image_2)
        
        ref = torch.hstack((ref_feat,ref_feat_2))
        dist = torch.hstack((dist_feat,dist_feat_2))
        feat = torch.abs(ref - dist)

        score = feat @ self.k
        return torch.mean(score)




class WResNet(nn.Module):
    def __init__(self, device = "cuda:0"):
        super().__init__()
        from argparse import ArgumentParser
        import torch
        from scipy import stats
        from torch import nn
        import torch.nn.functional as F
        from PIL import Image
        from RADN.main import RandomCropPatches, NonOverlappingCropPatches
        import numpy as np
        from RADN.model.WResNet import WResNet
        from RADN.model.RADN import RADN
        model_n = "WResNet"
        device = device
        if model_n == 'WResNet':
            self.model = WResNet().to(device)
        elif model_n == 'RADN':
            self.model = RADN().to(device)
        self.model.load_state_dict(torch.load('RADN/checkpoints/WResNet-lr=0.0001-bs=2.6360'), False)
        self.model.eval()
    
    def forward(self, X, Y):
        dist_patches, ref_patches = CropPatches(X, Y)
        score = self.model((dist_patches, ref_patches))
        return -torch.mean(score)


class MRperceptual(nn.Module):
    def __init__(self, device = "cuda:0", mode = 'mrpl'):
        super().__init__()
        spatial = False
        import MR_perceptual.mrpl as mrpl
        if mode == 'mrpl':
            self.loss_fn = mrpl.MRPL(net='alex', spatial=spatial,mrpl=True,verbose=0) 
        elif mode == 'mr_simple':
            self.loss_fn = mrpl.MRPL(net='alex', spatial=spatial, mrpl=False, loss_type='CE', norm='sigmoid', feature='linear', resolution=['x1','x2'], mrpl_like=True,verbose=0) 
        else :
            raise('Not implemented !')
    def forward(self, X, Y):
        #print(X.shape , Y.shape)
        val = self.loss_fn.forward(X, Y)#lower -- better
        return torch.mean(val)


class IQATransformerBNS(nn.Module):#<192
    def __init__(self, device = "cuda:0", mode = 'mrpl', crop_sz = 192):
        super().__init__()
        self.crop_sz = crop_sz
        import importlib
        import sys
        from  IQAConformerBNS.functions import load_model
        from IQAConformerBNS.pretrainedmodels import  inceptionresnetv2
        sys.path.append("./IQAConformerBNS/")
        #importlib.import_module("IQAConformerBNS.configs.PIPAL.IQA_Transformer")
        IQAConformerBNS = importlib.import_module("IQAConformerBNS.configs.PIPAL.IQA_Transformer")
        self.model = IQAConformerBNS.model
        self.model.load("IQAConformerBNS\pretrainedmodels\checkpoints_swa-equal-21-30.ckpt")

        import torch
        import torchvision.transforms
        device = "cuda:0"
        self.model = self.model.to(device)
    def forward(self, X, Y):
        val = self.model([X[...,:self.crop_sz,:self.crop_sz],\
            Y[...,:self.crop_sz,:self.crop_sz]])
        return -torch.mean(val)

class IQAConformerBNS(nn.Module):#<192
    def __init__(self, device = "cuda:0", mode = 'mrpl', crop_sz = 192):
        super().__init__()
        self.crop_sz = crop_sz
        import importlib
        import sys
        from  IQAConformerBNS.functions import load_model
        from IQAConformerBNS.pretrainedmodels import  inceptionresnetv2
        sys.path.append("./IQAConformerBNS/")
        #importlib.import_module("IQAConformerBNS.configs.PIPAL.IQA_Transformer")
        IQAConformerBNS = importlib.import_module("IQAConformerBNS.configs.PIPAL.IQA_Conformer")
        self.model = IQAConformerBNS.model
        self.model.load("IQAConformerBNS\callbacks\PIPAL\IQA_Conformer\checkpoints_swa-equal-21-30.ckpt")
        self.model = self.model.to(device)
    def forward(self, X, Y):
        val = self.model([X[...,:self.crop_sz,:self.crop_sz],\
            Y[...,:self.crop_sz,:self.crop_sz]])
        return -torch.mean(val)


import IQA_pytorch as iqa

class DISTScrops(nn.Module):
    def __init__(self, device = "cuda:0", mode = 'mrpl', crop_sz = 192, crops_num = 1):
        super().__init__()
        self.crop_sz = crop_sz
        self.crops_num = crops_num
        self.loss = iqa.DISTS().to(device)
    def forward(self, X, Y):
        X, Y = CropPatches(X,Y,self.crop_sz,self.crops_num)
        X,Y = X.reshape(X.shape[0]*X.shape[1],X.shape[2], X.shape[3], X.shape[4]), \
            Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4])
        val = self.loss(X, Y)
        return torch.mean(val)
