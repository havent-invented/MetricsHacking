

import pickle
import code
import os
from turtle import forward
import torch
import torch.nn
import torchvision.utils
import torchvision
class entropy_bottleneck:
    def __init__(self):
        self.loss = lambda : 0
def Zero_lambda():
    return 0

class codec_H264_real(torch.nn.Module):
    def __init__(self, q, compress_path = "R:/", in_img_nam = "in.png", out_img_nam = "out.png", comp_file_nam = "compressed_tmp.mp4", codec_opt = " ", contrast_adjust = "yes", x_hat_format = False):
        super(codec_H264_real, self).__init__()
        self.codec_opt = codec_opt
        self.q = q
        self.codec_opt += f" -b:v {self.q}k "
        self.in_img_nam = in_img_nam
        self.out_img_nam = out_img_nam 
        self.in_img_path = os.path.join(compress_path, in_img_nam)
        self.out_img_path = os.path.join(compress_path, out_img_nam)
        self.compressed_file = os.path.join(compress_path, comp_file_nam)
        self.contrast_adjust = contrast_adjust
        self.x_hat_format = x_hat_format
        if self.x_hat_format:
            self.X_hat = None
            with open(os.path.join('./sample_data/likelihoods.pkl'), 'rb') as f:
                self.X_hat = pickle.load(f)
            self.X_out = {"likelihoods": self.X_hat}
            self.entropy_bottleneck = entropy_bottleneck()
            self.entropy_bottleneck.loss = Zero_lambda
    def named_parameters(self):
        return {("3.quantiles",nn.Parameter(torch.tensor([[0.]]))) : nn.Parameter(torch.tensor([[0.]]))} 
    def forward(self, input):
        output = torch.zeros_like(input, device = input.device)
        for idx, input_X in enumerate(input):
            torchvision.utils.save_image(input_X, self.in_img_path)
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {self.in_img_path} {self.codec_opt} -pix_fmt yuv420p {self.compressed_file}")
            os.system(f"ffmpeg -hide_banner -loglevel error -y -i {self.compressed_file} {self.out_img_path}")
            output_X = torchvision.io.read_image(self.out_img_path)
            #print(f"output_X:{output_X}" )
            output_X = output_X.float() 
            output_X = (output_X - output_X.mean()) 
            if output_X.std() > 1e-7:
                 output_X = output_X / (output_X.std()+1e-05)
            output[idx, :] = output_X
            if self.contrast_adjust == 'no':
                pass
            elif self.contrast_adjust == 'yes':
                for ch in range(output.shape[1]):
                    output[idx, ch] = (output[idx, ch] - output[idx, ch].mean()) / (1e-05+output[idx, ch].std()) * input[idx, ch].std() + input[idx, ch].mean()
            #print(f"output_X_p:{output_X}")
        if self.x_hat_format:
            self.X_out['x_hat'] = output
            return self.X_out
        else:
            return output

            
"""
X = torch.randn(3,3,256,256)
codec = codec_H264_real()
Y = codec(X)

import numpy as np
PSNR = 20 * (np.log10(1) - 10 * np.log10((Y-X).abs().mean()))
print(PSNR)
codec.contrast_adjust = "no"
Y = codec(X)


import numpy as np
PSNR = 20 * (np.log10(1) - 10 * np.log10((Y-X).abs().mean()))
print(PSNR)
# PSNR on image """

