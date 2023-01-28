import pix2pixARCS.Generative_Adversarial_Perturbations.generators as generators
from pix2pixARCS.UnetViT.self_attention_cv.transunet import TransUnet
import pix2pixARCS.EdgeNeXt.models.model as modelsEdgeNeXt
import torch
import torch.nn as nn
from pix2pixARCS.SwinIR.runapi import api_link

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

enhance_Identity = enhance_Identity()

class smallnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(nn.Conv2d(3, 16, (3,3), padding = 1, padding_mode = 'reflect' ),
                nn.ReLU(inplace=True),)
        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),
            nn.Conv2d(16, 32, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),
            nn.Conv2d(32, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),)
        self.seq3 = nn.Sequential(
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),
            )
        self.seq4 = nn.Sequential(
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),
                nn.LeakyReLU(),)
        self.seq5 = nn.Sequential(nn.Conv2d(16, 3, (3,3), padding = 1, padding_mode = 'reflect'),)
        
    def forward(self, inputX):    
        x = self.seq1(inputX)
        x1 = x
        x = self.seq2(x) + x
        x = self.seq3(x) + x
        x = self.seq4(x) + x
        x = x1 + x
        x = self.seq5(x)
        return x

class smallnet_skips(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(nn.Conv2d(3, 16, (3,3), padding = 1, padding_mode = 'reflect'),#, padding="same"),
                nn.ReLU(inplace=True),)
        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 16, (3,3),padding = 1, padding_mode = 'reflect'),# padding="same"),
                nn.LeakyReLU(),
            nn.Conv2d(16, 32, (3,3), padding = 1, padding_mode = 'reflect'),# padding="same"),
                nn.LeakyReLU(),
            nn.Conv2d(32, 16, (3,3), padding = 1, padding_mode = 'reflect'), #padding="same"),
                nn.LeakyReLU(),)
        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 16, (3,3), padding = 1, padding_mode = 'reflect'), #padding="same"),
                nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'),# padding="same"),
                nn.LeakyReLU(),
            )
        self.seq4 = nn.Sequential(
            nn.Conv2d(48, 16, (3,3), padding = 1, padding_mode = 'reflect'), #padding="same"),
                nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3,3), padding = 1, padding_mode = 'reflect'), #padding="same"),
                nn.LeakyReLU(),
            nn.Conv2d(16, 16, (3,3),padding = 1, padding_mode = 'reflect'), #padding="same"),
                nn.LeakyReLU(),)
        self.seq5 = nn.Sequential(nn.Conv2d(80, 3, (3,3),padding = 1, padding_mode = 'reflect')) #padding="same"),)
        
    def forward(self, inputX):    
        x = self.seq1(inputX)
        x1 = x
        x = torch.cat([x, self.seq2(x)], 1)
        x = torch.cat([x, self.seq3(x)], 1)
        x = torch.cat([x, self.seq4(x)], 1)
        x = torch.cat([x, x1], 1)
        #print(self.seq5)
        x = self.seq5(x)
        return x

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
        #def add_pad(X):
        #    return torch.nn.functional.pad(X,pad = (0,64,0,64), mode = 'reflect')
        #def cut_pad(X):
        #    return X[...,:-64, : -64]
        
        
        #input = torchvision.transforms.Lambda(add_pad)(input) 
        
        input_shape = None
        if input.shape[-1] % 32 or input.shape[-2] %32:
            input_shape = input.shape
            input = torch.nn.functional.pad(input,pad = (0,(32-input.shape[-1]%32)%32,0,(32-input.shape[-2]%32)%32 ), mode = 'reflect')
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
        if input_shape != None:
            out = out[..., :input_shape[-2], :input_shape[-1]]
        #out = torchvision.transforms.Lambda(cut_pad)(out)
        return out

def get_enh_model(enhance_net, cfg = None):
    model = None
    if enhance_net == "GAPresnet":#Tanh
        model = generators.define(3,3,15,"resnet", gpu_ids=[0])
    elif enhance_net == "GAPunetrec":
        model = generators.define(3,3,15,'unet-rec', gpu_ids=[0])
    elif enhance_net == "GAPunetsc":
        model = generators.define(3,3,15,'unet-sc', gpu_ids=[0])
    elif enhance_net == "GAPunet":
        model = generators.define(3,3,15,'unet', gpu_ids=[0])
    elif enhance_net == "TransUnet":#Conv2d
        model = TransUnet(in_channels=3, img_dim=256, vit_blocks=8, vit_dim_linear_mhsa_block=512, classes=3)#128
    elif enhance_net == "EdgeNeXtbase":#classification task only((
        model = modelsEdgeNeXt.edgenext_base( pretrained=True, **{"classifier_dropout": 0.0})
        model.head = nn.Linear(584, 3, bias=True)
    elif enhance_net == "EdgeNeXtxxsmall":
        model = modelsEdgeNeXt.edgenext_xx_small( pretrained=True, **{"classifier_dropout": 0.0})
        model.head = nn.Linear(584, 3, bias=True)
    elif enhance_net == "EdgeNeXtxsmall":
        model = modelsEdgeNeXt.edgenext_x_small(**{"classifier_dropout": 0.0}, pretrained=True)
        model.head = nn.Linear(584, 3, bias=True)
    elif enhance_net == "EdgeNeXtsmallbnhs":
        model = modelsEdgeNeXt.edgenext_small_bn_hs(**{"classifier_dropout": 0.0}, pretrained=True)
        model.head = nn.Linear(584, 3, bias=True)
    elif enhance_net == "SwinIR_sr1":#Conv2d
        model = api_link("sr1")
    #elif enhance_net == "SwinIR_sr2x":
        #model = api_link("sr2x")
    elif enhance_net == "SwinIR_jpeg":#Conv2d
        model = api_link("jpeg")
        

    elif enhance_net == "Resnet18Unet":
        model = ResNetUNet(3)
    elif cfg["general"]["enhance_net"] == "smallnet":
        model = smallnet()
    elif cfg["general"]["enhance_net"] == "smallnet_skips":
        model = smallnet_skips()
    elif cfg["general"]["enhance_net"] == "No" or cfg["general"]["enhance_net"] == "Identity": 
        model = enhance_Identity
    elif cfg["general"]["enhance_net"] == "mobile_deepest_resnet24":
        sys.path.append(os.path.join(cfg['general']['project_dir'], "OMGD"))
        model = torch.load(os.path.join(cfg['general']['project_dir'], "OMGD/m24.ckpt"))
    if cfg != None:
        model = model.to(cfg["run"]["device"]).requires_grad_(True)
        for i in model.parameters():
            i.retain_grad()
    
    return model
    


