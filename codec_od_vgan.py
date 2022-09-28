import sys
import torch.nn as nn 
import sys
import torch
sys.path.append("./DiffJPEG16/")
sys.path.append("../OD_VGAN/")
from DiffJPEG16 import DiffJPEG as DiffJPEG_c
import os
class entropy_bottleneck:
    def __init__(self):
        self.loss = lambda : 0
def Zero_lambda():
    return 0
class codec_ODVGAN(nn.Module):
    def __init__(self, cfg):
        import torchvision
        super().__init__()
        import pickle
        self.cfg = {"general": cfg['general']}
        self.convert_f = torch.load(cfg["general"]["od_vgan_model_path"]).to(cfg["run"]["device"])      
        self.X_hat = None
        with open(os.path.join(self.cfg["general"]["project_dir"], 'sample_data/likelihoods.pkl'), 'rb') as f:
            self.X_hat = pickle.load(f)
        self.X_out = {"likelihoods": self.X_hat}
        self.entropy_bottleneck = entropy_bottleneck()
        self.entropy_bottleneck.loss = Zero_lambda
    def named_parameters(self):
        return {("3.quantiles",nn.Parameter(torch.tensor([[0.]]))) : nn.Parameter(torch.tensor([[0.]]))} 
    def forward(self, X):
        self.X_out['x_hat'] = self.convert_f(X)
        return self.X_out
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self
