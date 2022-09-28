

import pickle
import code
import os
from turtle import forward
import torch
import torch.nn
import torchvision.utils
import torchvision.transforms as T
from PIL import Image
import torchvision
class entropy_bottleneck:
    def __init__(self):
        self.loss = lambda : 0
def Zero_lambda():
    return 0

class codec_jpeg_real(torch.nn.Module):
    def __init__(self, q : int, compress_path = "R:/", in_img_nam = "in.png", out_img_nam = "out.png", comp_file_nam = "compressed_tmp.jpeg", contrast_adjust = "simple", x_hat_format = False, mode = 0):
        super(codec_jpeg_real, self).__init__()
        self.q = q
        self.mode = mode
        self.img_idx = 0
        self.in_img_nam = in_img_nam
        self.out_img_nam = out_img_nam 
        self.in_img_path = os.path.join(compress_path, in_img_nam)
        self.out_img_path = os.path.join(compress_path, out_img_nam)
        self.compressed_file = os.path.join(compress_path, comp_file_nam)
        self.contrast_adjust = contrast_adjust
        self.x_hat_format = x_hat_format
        self.bpp = 1
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
        if self.mode == 0:
            for idx, input_X in enumerate(input):
                torchvision.utils.save_image(input_X, self.in_img_path)
                os.system(f"ffmpeg -hide_banner -loglevel error -y -i {self.in_img_path} -q:v {self.q} {self.compressed_file}")
                os.system(f"ffmpeg -hide_banner -loglevel error -y -i {self.compressed_file} {self.out_img_path}")
                self.file_sz = os.path.getsize(self.compressed_file) / 1e6
                output_X = torchvision.io.read_image(self.out_img_path)
                #print(f"output_X:{output_X}" )
                output_X = output_X.float() 
                output_X = (output_X - output_X.mean()) 
                if output_X.std() > 1e-7:
                     output_X = output_X / (output_X.std()+1e-05)
                output[idx, :] = output_X
        elif self.mode == 1:
            self.bpp = 0
            file_sz_sum = 0
            pixels_sum = 0
            for idx, input_X in enumerate(input):
                pil_img = T.ToPILImage()(input_X)
                pil_img.save(self.compressed_file, format = 'jpeg', quality = self.q)
                output_X = torchvision.io.read_image(self.compressed_file)
                file_sz_sum = 8 * os.path.getsize(self.compressed_file) 
                if len(input_X.shape) == 3:
                    pixels_sum = input_X.shape[0] * input_X.shape[1] * input_X.shape[2]
                else:
                    pixels_sum = input_X.shape[0] * input_X.shape[1]
                self.img_idx += 1
                output[idx, :] = output_X
                 
                self.bpp += file_sz_sum / pixels_sum
            self.bpp /= len(input)
        elif self.mode == 2:
            import cv2
            self.bpp = 0 
            for idx, input_X in enumerate(input):
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.q]
                result, encimg = cv2.imencode('.jpg', (input_X*255).cpu().numpy().swapaxes(0,2).swapaxes(0,1), encode_param)
                decimg = cv2.imdecode(encimg, 1)
                #print(result)
                file_sz_sum =len(encimg)
                #print(f"decimg:{decimg}")
                if len(input_X.shape) == 3:
                    pixels_sum = input_X.shape[0] * input_X.shape[1] * input_X.shape[2]
                else:
                    pixels_sum = input_X.shape[0] * input_X.shape[1]
                output_X = torch.from_numpy(decimg.swapaxes(0,1).swapaxes(0,2)) #/ 255. 
                #print(f"output_X:{output_X}")
                output[idx, :] = output_X 
                self.bpp += file_sz_sum / pixels_sum
            self.bpp /= len(input)

        if self.contrast_adjust == 'no':
            pass
        elif self.contrast_adjust == 'yes':
            for ch in range(output.shape[1]):
                output[idx, ch] = (output[idx, ch] - output[idx, ch].mean()) / (1e-05+output[idx, ch].std()) * input[idx, ch].std() + input[idx, ch].mean()
                #print(f"output_X_p:{output_X}")
        elif self.contrast_adjust == 'simple':
            if output.mean() > 3:
                output = output / 255. 
            output = torch.clamp(output, 0., 1.)
        if self.x_hat_format:
            self.X_out['x_hat'] = output
            if self.mode == 1 or self.mode == 0 or self.mode == 2:
                self.X_out['bpp_loss'] = torch.tensor(self.bpp)
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

