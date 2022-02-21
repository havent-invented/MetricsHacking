import torchvision
import torch

class enhance_Identity():
    def __init__(self):
        pass
    def named_parameters(self):
        return {("3.quantiles",torch.nn.Parameter(torch.tensor([[0.]]))) : torch.nn.Parameter(torch.tensor([[0.]]))} 
    def parameters(self):
        return self.named_parameters() 
    def forward(self, X):
        return X
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self

class codec_Identity():
    def __init__(self):
        import pickle
        self.X_hat = None
        with open('./sample_data/likelihoods.pkl', 'rb') as f:
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

#datalen_train = 4#128
#datalen_test = 4#32

optimize_image = False
net_enhance = enhance_Identity() if optimize_image else None
net_codec = codec_Identity() if optimize_image else None
save_netcodec = False
save_net_enhance = True




import sys
sys.path.insert(1, "E:/VMAF_METRIX/NeuralNetworkCompression/")
exec(open('main.py').read())#MAIN
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
patch_sz = 256

#try:
#    save_filename
#except Exception:
save_filename = "vimeo11k_PAC2PIC_20mse_enhance_cheng2020_attn_quality2_fixed_direction"
    #save_filename = "vimeo11k_Linearity_2000mse_enhance_no_codec_fixed_direction"
try:
    device
except Exception:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    save_netcodec
except Exception:
    save_netcodec = False
try:
    save_net_enhance
except Exception:
    save_net_enhance = True
try:
    net_enhance
except Exception:
    net_enhance = None
X = None
try:
    optimize_image
except Exception:
    optimize_image = False
try:
    loss_calc
except Exception:
    loss_calc = None
try:
    net_codec
except Exception:
    net_codec = None
try:
    datalen_train
except Exception:
    datalen_train = 11000
try:
    datalen_test
except Exception:
    datalen_test = 400

class enhance_Identity():
    def __init__(self):
        pass
    def named_parameters(self):
        return {("3.quantiles",torch.nn.Parameter(torch.tensor([[0.]]))) : torch.nn.Parameter(torch.tensor([[0.]]))} 
    def parameters(self):
        return self.named_parameters() 
    def forward(self, X):
        return X
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self

class codec_Identity():
    def __init__(self):
        import pickle
        self.X_hat = None
        with open('./sample_data/likelihoods.pkl', 'rb') as f:
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
    
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

import torch
import os
import numpy as np
import random
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import h5py

def Linearity_met(im, device = device):
    
    model = IQAModel().to(device)  #
    im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

    checkpoint = torch.load("E:/VMAF_METRIX/NeuralNetworkCompression/LinearityIQA/LinearityIQA/../p1q2.pth")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    y = model(im.unsqueeze(0))
    k = checkpoint['k']
    b = checkpoint['b']
    print('The image quality score is {}'.format(y[-1].item() * k[-1] + b[-1]))


class koniq(nn.Module):
    def __init__(self, model_dir ="E:/VMAF_METRIX/NeuralNetworkCompression/koniq/", device = device):
        super().__init__()
        import sys
        sys.path.insert(1, model_dir)
        from inceptionresnetv2 import inceptionresnetv2
        class model_qa(nn.Module):
            def __init__(self,num_classes,**kwargs):
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
        self.KonCept512 = self.KonCept512.to(device)
        
    def forward(self, im, device = device):
        #patch size must be >= (299,299)
        out = self.KonCept512(im).mean()
        return out
   
    
import torch.nn as nn
from torchvision.transforms.functional import resize, to_tensor, normalize
class Linearity(nn.Module):
    def __init__(self, model_dir = "E:/VMAF_METRIX/NeuralNetworkCompression/LinearityIQA/LinearityIQA/", device = device):
        super().__init__()
        sys.path.insert(1, model_dir)
        from IQAmodel import IQAModel
        self.model = IQAModel().to(device)
        checkpoint = torch.load(model_dir +"../p1q2.pth")
        self.k = checkpoint['k']
        self.b = checkpoint['b']
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(device)
        del checkpoint
        
    def forward(self, im, device = device):
        y = self.model(im)
        val = (y[-1]* self.k[-1] + self.b[-1]).mean()
        return val / 100.
    
class NIMA(nn.Module):
    def __init__(self, model_dir ="E:/VMAF_METRIX/NeuralNetworkCompression/Neural-IMage-Assessment/", device = device):
        super().__init__()
        import sys
        import torchvision.models as models
        import torchvision.transforms as transforms
        sys.path.insert(1, model_dir)
        from model.model import NIMA   
        self.base_model = models.vgg16(pretrained=True).to(device)
        self.model = NIMA(self.base_model).to(device)
        self.model.load_state_dict(torch.load(model_dir +"model/epoch-82.pth"))
    def forward(self, im, device = device):
        out = self.model(im[:,:,:224,:224]).mean()
        return out

class VSFA_loss(nn.Module):
    def __init__(self, model_dir = "E:/VMAF_METRIX/NeuralNetworkCompression/VSFA/VSFA/", device = device):
        super().__init__()
        import sys
        sys.path.insert(1, model_dir)
        import VSFA
        from CNNfeatures import get_features
        self.get_features = get_features
        self.model = VSFA.VSFA()
        self.model.load_state_dict(torch.load(model_dir + "models/VSFA.pt"))
        self.model.to(device)
    def forward(self, X_sample, device = device):
        self.features = self.get_features(X_sample, frame_batch_size = len(X_sample), device=device)
        self.features = torch.unsqueeze(self.features, 0)  # batch size 1
        input_length = self.features.shape[1] * torch.ones(1, 1)
        outputs = self.model(self.features, input_length)
        
        return outputs[0][0]
from piq import PieAPP
class BRISQ(nn.Module):
    def __init__(self):
        super().__init__()
        from piq import BRISQUELoss
        self.model = BRISQUELoss()
    def forward(self, X_sample):
        val = self.model(torch.clamp(X_sample,0,1))
        return val

class SPAQ(nn.Module):
    def __init__(self, model_dir = "E:/VMAF_METRIX/NeuralNetworkCompression/SPAQ", device = device):
        super().__init__()
        sys.path.insert(1, model_dir)
        from BL_demo import Demo#Changed map_location
        self.dm = Demo("", checkpoint_dir='E:/VMAF_METRIX/NeuralNetworkCompression/SPAQ/weights/BL_release.pt', device = device )
        self.dm.model = self.dm.model.to(device)
    def forward(self, im, device = device):
        score_1 = self.dm.model(im).mean()
        return score_1 / 100.

class paq2pic_model(nn.Module):
    def __init__(self, model_dir = "E:/VMAF_METRIX/NeuralNetworkCompression/paq2piq/", device = device, blk_size = None):
        super().__init__()
        import sys
        sys.path.insert(1,model_dir)
        from paq2piq_standalone import InferenceModel, RoIPoolModel
        self.model = InferenceModel(RoIPoolModel(backbone='resnet18', pretrained=True), model_dir + "models/RoIPoolModel-fit.10.bs.120.pth")
        if blk_size != None:
            self.model.blk_size = blk_size
        
    def forward(self, X_sample, device = device):
        batch_sz = len(X_sample)
        global_score_batch = 0
        for X_i in X_sample:
            t = self.model.model(X_i.unsqueeze(0))[0]
            self.model.model.input_block_rois(self.model.blk_size, [X_sample.shape[-2], X_sample.shape[-1]], device=device)
            global_score = t[0]
            global_score_batch += global_score
        global_score_batch = global_score_batch / batch_sz /100.
        return global_score_batch
    
class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
if net_enhance == None:
    net_enhance = ResNetUNet(3).to(device)

#net_codec = bmshj2018_factorized(quality=2, pretrained=True).train().to(device)
#mbt2018
if net_codec == None:
    net_codec = cheng2020_attn(quality=2, pretrained=True).train().to(device)# ssf2020 -- video
env = calc_met( model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir)
#env.datagen = [frameGT for frameGT in skvideo.io.FFmpegReader(env.dataset_dir + env.dataset[0], outputdict={"-c:v" :" rawvideo","-f": "rawvideo"}).nextFrame()]
self = env



from piq import LPIPS as piq_LPIPS#PieAPP VSI, FSIM, NLPD, deepIQA
from piq import DISTS as piq_DISTS
import IQA_pytorch as iqa#SSIM, GMSD, LPIPSvgg, DISTS
class calc_met:
    def __init__(self,dataset1 = ["Run439.Y4M"], convKer1 = None, home_dir1 = "R:/", creat_dir = False, calc_SSIM_PSNR = False, calc_model_features = False,device = device, model = "vmaf_v063" , codec = '   -preset:v medium -x265-params log-level=error ',dataset_dir = "dataset/"):
        self.device = device
        self.model = VQAModel().to(device)
        self.model.load_state_dict(torch.load('../models/MDTVSFA.pt'))
        self.model.train()
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
        with torch.enable_grad():
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

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["PSNR"] = 10 * torch.log10(1/ out["mse_loss"])
        out["loss_classic"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out

class Custom_enh_Loss(nn.Module):
    def __init__(self, lmbda=1e-2, device = device):
        super().__init__()
        self.rdLoss = RateDistortionLoss(lmbda)
        #self.lpips = iqa.LPIPSvgg().to(device)
        #self.ssim = iqa.SSIM()
        #self.dists = iqa.DISTS().to(device)
        #self.MDTVSFA_metr = calc_met()
        #self.brisq_loss = BRISQ()   
        #self.lin_loss = Linearity()
        #self.lin_loss.requires_grad_()
        #self.spaq_loss = SPAQ()
        #self.vsfa_loss = VSFA_loss()
        #piapp_loss = PieAPP()
        self.paq2pic_loss = paq2pic_model()
        #self.NIMA_loss = NIMA()
        #self.koniq_loss = koniq()
    def forward(self, X_out, Y):
        if X_out['x_hat'].device != Y.device:
            X_out['x_hat'] = X_out['x_hat'].to(device)
        self.loss = self.rdLoss(X_out, Y)
        #self.loss['MDTVSFA'] = -self.MDTVSFA_metr.MDTVSFA(X_out['x_hat'])
        #self.loss["DISTS"] = self.dists(X_out['x_hat'], Y)
        #self.loss["LPIPS"] = self.lpips(X_out['x_hat'], Y)
        lmbda = 1e-2
        #self.loss["SSIM"] = self.ssim(X,X_out['x_hat'])
        
        #self.loss["BRISQ"] = self.brisq_loss(X_out['x_hat'])
        #self.loss["Linearity"] = -self.lin_loss(X_out['x_hat'])
        #self.loss["SPAQ"] = -self.spaq_loss(X_out['x_hat'])
        self.loss["PAC2PIC"] = -self.paq2pic_loss(X_out['x_hat'])
        #self.loss["NIMA"] = -self.NIMA_loss(X_out['x_hat'])
        #self.loss["KONIQ"] = -self.koniq_loss(X_out['x_hat'])
        
        
        #self.loss["VSFA"] = -self.vsfa_loss(X_out['x_hat'])
        self.loss["loss"] = self.loss["PAC2PIC"] + 20*self.loss["mse_loss"]
        #+ 2000*self.loss["mse_loss"] #self.loss["Linearity"] +200* 
        #self.loss["loss"] = self.loss["SPAQ"] + 2000*self.loss["mse_loss"]#+ 2000*self.loss["mse_loss"] #self.loss["Linearity"] +200* self.loss["mse_loss"]  #+ loss["DISTS"] +  loss['MDTVSFA'] #+ loss["bpp_loss"] + lmbda / 2 * loss["mse_loss"] * 255 ** 2# * loss["mse"] + loss["bpp_loss"]
        #loss["aux_loss"] = net_codec.aux_loss()
        return self.loss
if loss_calc == None:
    loss_calc = Custom_enh_Loss()
class Video_reader_read():
    def __init__(self,name1 = dst_dir + "blue_hair_1920x1080_30.yuv.Y4M"):
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
#X_orig = cheng2020_attn(quality=2, pretrained=True)(X.cpu())

rdLoss = RateDistortionLoss()
from torch.utils.data import Dataset, IterableDataset
dst_dir_vimeo = 'P:/vimeo_triplet/sequences/'
from torchvision.io import read_image
from torch.utils.data import DataLoader
import os
import torchvision
def dir_of_dirs(paths):
    A = []
    for j in paths:
        for i in os.listdir(j):
            A.append(os.path.join(j, i))
    return A

class Video_reader_dataset(Dataset):
    def __init__(self, all_frames = True, num_frames = 10, name1 = dst_dir + "blue_hair_1920x1080_30.yuv.Y4M"):
        super(CustomImageDataset).__init__()
        self.nameGT = name1
        self.temp_reader1 = skvideo.io.FFmpegReader(self.nameGT, outputdict={"-c:v" :" rawvideo","-f": "rawvideo"})
        self.datagenGT = [frameGT / 255. for frameGT in self.temp_reader1.nextFrame()]
        self.temp_reader1.close()
        self.datagenGT = np.array([[i[:,:,0],i[:,:,1],i[:,:,2]] for i in self.datagenGT])
        if all_frames:
            self.lst_1 = torch.tensor(self.datagenGT).float() 
        else:
            self.lst_1 = torch.tensor(self.datagenGT[: num_frames]).float() 
        self.datalen = len(self.lst_1)
    def __len__(self):
        return self.datalen
    def __getitem__(self, idx):
        return self.lst_1[idx]

    
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None,train = True, datalen = 128):
        super(CustomImageDataset).__init__()
        self.datalen = datalen
        self.train = train
        self.image = 0
        self.label = 0
        self.img_names = dir_of_dirs(dir_of_dirs(dir_of_dirs([dst_dir_vimeo])))
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
        return torchvision.transforms.RandomResizedCrop((patch_sz, patch_sz))(self.image) / 255.
    
#dataset = CustomImageDataset(dst_dir_vimeo)#219k
#dataset_train = iter(DataLoader(dataset, batch_size= 16, shuffle = True))#13k

#dataset_train, dataset_test = torch.utils.data.random_split( dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)])
dataset_train = CustomImageDataset(dst_dir_vimeo,train= True, datalen = datalen_train)
dataset_test = CustomImageDataset(dst_dir_vimeo,train= False, datalen = datalen_test)
dataset_train = DataLoader(dataset_train, batch_size= 8, shuffle = True)#8#4#8
dataset_test = DataLoader(dataset_test, batch_size= 2, shuffle = True)#8#4#4
mse_loss = nn.MSELoss()
#opt_target = [i for i in net_codec.parameters()]
opt_target = [p for n,p in net_codec.named_parameters()]
#optimizer = optim.Adam(opt_target, lr = 0.001)
dr_name = "v_mse"
AV_log = []
curve_mse = []
plot_data = []
plot_data_mse = []
from IPython.display import clear_output
if optimize_image:
    X = next(iter(dataset_train)).detach().to(device)
    Y = X.detach().to(device)
    X.requires_grad_()
    X.retain_grad()
parameters = set(p for n, p in net_enhance.named_parameters()) if not optimize_image else [X]
aux_parameters = set(p for n, p in net_codec.named_parameters() if n.endswith(".quantiles"))
aux_loss = net_codec.entropy_bottleneck.loss()

optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)


save_result = True
X_sample = torch.load("sample_data/X.ckpt").to("cpu")

n = 30
logs_plot_cur = {}
logs_plot = {}
max_epoch = 12
skip_0epoch = True
for epoch in tqdm(range(max_epoch)):
    idx_video = 0
    logs_plot_cur = {}
    if skip_0epoch and epoch == 0:
        continue
    for to_train in [True, False]:
        tqdm_dataset = tqdm(dataset_train if to_train else dataset_test)
        for frame in tqdm_dataset:
            idx_video += 1
            if not optimize_image:
                X = frame
                X = torchvision.transforms.RandomResizedCrop((patch_sz,patch_sz))(X)
                X = X.to(device)#X = X.detach().to(device)
                Y = X.detach().clone().to(device)
            X.data.clamp_(min=0,max=1)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            X_enhance = net_enhance(X)
            X_enhance.data.clamp_(min=0,max=1)
            X_out = net_codec.forward(X_enhance)
            X_out['x_hat'].data.clamp_(min=0,max=1)
            
            
            #X_out['x_hat'] = torch.nan_to_num(X_out['x_hat'])
            #Y = torch.nan_to_num(Y)
            
            loss = loss_calc(X_out, Y)
            
            lmbda = 1e-2
            if epoch != 0 and to_train:
                loss["loss"].backward()
                #list(parameters)[0].grad = torch.nan_to_num(list(parameters)[0].grad)
                #list(parameters)[0] = torch.nan_to_num(list(parameters)[0])
                optimizer.step()
            #loss["aux_loss"] = net_codec.aux_loss()
            #if epoch != 0 and to_train:
                #loss["aux_loss"].backward()
                #aux_optimizer.step()
                
            for j in list(loss.keys()):
                j_converted = j + ("_test" if not to_train else "")
                if not j_converted in logs_plot_cur:
                    logs_plot_cur[j_converted] = []
                logs_plot_cur[j_converted].append(loss[j].data.to("cpu").numpy())
            
            #X_enhance.data.clamp_(min=0,max=1)
            #X.data.clamp_(min=0,max=1)
            #X_out['x_hat'].data.clamp_(min=0,max=1)
            #torch.nn.utils.clip_grad_norm_(parameters, 1)
            
    if not to_train:
        for j in list(logs_plot_cur.keys()):
            if not j in logs_plot:
                logs_plot[j] = []
            logs_plot[j].append(np.mean(logs_plot_cur[j]))
    if 1:
        clear_output()
        fig = plt.figure(figsize=(20,8))
        unique_names = list(np.unique(list(map(lambda x : x.split("_test")[0], list(logs_plot.keys())))))
        unique_idx = {i: j for i, j in zip(unique_names, range(len(unique_names)))}
        for plot_idx, plot_i in enumerate(list(logs_plot.keys())):
            short_name = plot_i.split("_test")[0]
            cur_idx = unique_idx[short_name]    
            plt.subplot(2, math.ceil( (len(unique_idx)+1) / 2), cur_idx + 1, )
            plt.xlabel("step_number")
            plt.title("Learning curve " + short_name)
            plt.ylabel(short_name)
            plt.plot(np.arange(len(logs_plot[plot_i])), logs_plot[plot_i], label = ("train" if short_name == plot_i else "test"))    
            plt.legend()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            plt.tight_layout() 
        if save_result == True:
            fig.savefig("vis/lerningcurve" + save_filename + ".png")
            if save_netcodec == True:
                torch.save(net_codec.state_dict(), "models/model_" +save_filename + ".ckpt") 
            if save_net_enhance and net_enhance != None:
                torch.save(net_enhance.state_dict(), "models_enhancement/model_" +save_filename + ".ckpt") 
            import pickle
            with open('logs_enhancement/plots'+ save_filename + '.pkl', 'wb') as f:
                pickle.dump(logs_plot, f)
    
        tqdm_dataset.refresh()
        plt.show()
        plt.figure(25)
        #X.data = X_sample.data
        #X_out = net_codec.forward(X)
        pltimshow_batch([Y, X_enhance, X_out['x_hat']], filename = "vis/pics_" + save_filename + ".png")
        plt.pause(0.005)