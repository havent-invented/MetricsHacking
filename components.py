

from Current_model_lib import *
def get_optim(cfg, parameters):
    if cfg["general"]["optimizer"] == "AdamW":
        cfg["run"]["optimizer"] = optim.AdamW(parameters, **cfg["general"]["optimizer_opt"])
    elif cfg["general"]["optimizer"] == "Adam":
        cfg["run"]["optimizer"] = optim.Adam(parameters, **cfg["general"]["optimizer_opt"])
    elif cfg["general"]["optimizer"] == "SGD":
        cfg["run"]["optimizer"] = optim.SGD(parameters, **cfg["general"]["optimizer_opt"])
    return cfg["run"]["optimizer"]

def get_enhance_net(cfg):    
    if cfg["general"]["enhance_net"] == "Resnet18Unet":
        cfg["run"]["net_enhance"] = ResNetUNet(3).to(cfg["run"]["device"]) 
        #cfg["run"]["net_enhance"] = nn.Sequential(nn.Conv2d(3, 3, 3,1, "same"),)
        cfg["run"]["net_enhance"] = cfg["run"]["net_enhance"].to(cfg["run"]["device"]).requires_grad_(True)
        for i in cfg["run"]["net_enhance"].parameters():
                i.retain_grad()
    elif cfg["general"]["enhance_net"] == "smallnet":
            cfg["run"]["net_enhance"] = smallnet().to(cfg["run"]["device"])
            cfg["run"]["net_enhance"] = cfg["run"]["net_enhance"].to(cfg["run"]["device"]).requires_grad_(True)
            for i in cfg["run"]["net_enhance"].parameters():
                i.retain_grad()
    elif cfg["general"]["enhance_net"] == "smallnet_skips":
            cfg["run"]["net_enhance"] = smallnet_skips().to(cfg["run"]["device"])
            cfg["run"]["net_enhance"] = cfg["run"]["net_enhance"].to(cfg["run"]["device"]).requires_grad_(True)
            for i in cfg["run"]["net_enhance"].parameters():
                i.retain_grad()
    elif cfg["general"]["enhance_net"] == "No" or cfg["general"]["enhance_net"] == "Identity": 
        cfg["run"]["net_enhance"] = enhance_Identity
    elif cfg["general"]["enhance_net"] == "mobile_deepest_resnet24":
        sys.path.append(os.path.join(cfg['general']['project_dir'], "OMGD"))
        cfg["run"]["net_enhance"] = torch.load(os.path.join(cfg['general']['project_dir'], "OMGD/m24.ckpt")).to(cfg["run"]["device"]).requires_grad_(True)
    return cfg["run"]["net_enhance"]
    
def get_codec(cfg, src_key = 'codec', target_key = "net_codec"):
    if cfg["general"][src_key] == "No":
        codec = codec_Identity
    elif cfg["general"][src_key] == "Blur":
        codec = codec_Blur(cfg, sigma = (cfg["general"]["blur_sigma_min"], cfg["general"]["blur_sigma_max"]), kernel_sizes = (cfg["general"]["blur_sz_min"], cfg["general"]["blur_sz_max"])).to(cfg["run"]["device"])
    elif cfg["general"][src_key] == "jpeg":
        from codec_jpeg import codec_JPEG
        codec = codec_JPEG(cfg).to(cfg["run"]["device"])
    elif cfg["general"][src_key] == "jpeg16":
        from codec_jpeg_16 import codec_JPEG
        codec = codec_JPEG(cfg).to(cfg["run"]["device"])
    elif cfg['general'][src_key] == "ODVGAN":
        sys.path.append("../OD_VGAN/")
        #codec = torch.load(cfg["general"]["od_vgan_model_path"]).to(cfg["run"]["device"])      
        from codec_od_vgan import codec_ODVGAN 
        codec = codec_ODVGAN(cfg).to(cfg["run"]["device"])
    elif cfg['general'][src_key] == "jpeg_real_ffmpeg":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        os.system(cfg['general']['home_dir'])
        codec = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 0, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general'][src_key] == "jpeg_real_pil":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        codec = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 1, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general'][src_key] == "jpeg_real_cv2":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        codec = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 2, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general'][src_key] == "img_h264_real":
        from compress_H264 import codec_H264_real
        codec = codec_H264_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'] , x_hat_format = True).to(cfg["run"]["device"])
    else:
        codec = cheng2020_attn(quality = cfg["general"]["quality"], pretrained = True, metric = cfg["general"]["codec_metric"]).to(cfg["run"]["device"]).requires_grad_(True)# ssf2020 -- video
    if target_key is not None:
        cfg["run"][target_key] = codec
    return codec




