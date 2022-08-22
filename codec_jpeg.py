import sys
import torch.nn as nn 
sys.path.append("./DiffJPEG/")
from DiffJPEG import DiffJPEG
import os

class codec_JPEG(nn.Module):
    def __init__(self, cfg):
        import torchvision
        super().__init__()
        import pickle
        self.cfg = {"general": cfg['general']}
        self.convert_f = DiffJPEG(cfg['general']['patch_sz'], cfg['general']['patch_sz'], quality = cfg["general"]["quality"]).to(cfg["run"]['device'])
        self.X_hat = None
        with open(os.path.join(self.cfg["general"]["project_dir"], 'sample_data/likelihoods.pkl'), 'rb') as f:
            self.X_hat = pickle.load(f)
        self.X_out = {"likelihoods": self.X_hat}
        class entropy_bottleneck:
            def __init__(self):
                self.loss = lambda : 0
        self.entropy_bottleneck = entropy_bottleneck()
        self.entropy_bottleneck.loss = lambda : 0
    def named_parameters(self):
        return {("3.quantiles",nn.Parameter(torch.tensor([[0.]]))) : nn.Parameter(torch.tensor([[0.]]))} 
    def forward(self, X):
        self.X_out['x_hat'] = self.convert_f(X)
        return self.X_out
    def __call__(self, X):
        return self.forward(X)
    def to(self, device):
        return self