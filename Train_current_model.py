import sys
from Current_model_lib import *
import matplotlib
matplotlib.use('Agg', force=True)#QtAgg

def train(cfg):
    if cfg["general"]["use_wandb"]:
        import wandb
        wandb.init(project=cfg["general"]["project_name"], entity="havent_invented", name = cfg["general"]["name"], tags = {"Train"}, save_code = True)
        wandb.config.update({"general": cfg["general"]})
    
    if not os.path.exists(cfg["general"]["home_dir"]):
        os.makedirs(cfg["general"]["home_dir"])

    
    X = None
    cfg["run"]["loss_calc"] = Custom_enh_Loss(target_lst = cfg["general"]["met_names"], k_lst = cfg ["general"] ["k_lst"]).eval().requires_grad_(False).to(cfg["run"]["device"])
   
    from enh_models import get_enh_model
    cfg["run"]["net_enhance"] = get_enh_model(cfg["general"]["enhance_net"], cfg)


    if "pretrained_model_path" in cfg["general"]:
        ckpt = torch.load(cfg["general"]['pretrained_model_path'])['model']
        cfg["run"]["net_enhance"].load_state_dict(ckpt)
                
    if cfg["general"]["codec"] == "No":
        cfg["run"]["net_codec"] = codec_Identity
    elif cfg["general"]["codec"] == "Blur":
        cfg["run"]["net_codec"] = codec_Blur(cfg, sigma = (cfg["general"]["blur_sigma_min"], cfg["general"]["blur_sigma_max"]), kernel_sizes = (cfg["general"]["blur_sz_min"], cfg["general"]["blur_sz_max"])).to(cfg["run"]["device"])
    elif cfg["general"]["codec"] == "jpeg":
        from codec_jpeg import codec_JPEG
        cfg["run"]["net_codec"] = codec_JPEG(cfg).to(cfg["run"]["device"])
    elif cfg["general"]["codec"] == "jpeg16":
        from codec_jpeg_16 import codec_JPEG
        cfg["run"]["net_codec"] = codec_JPEG(cfg).to(cfg["run"]["device"])
    elif cfg['general']['codec'] == "ODVGAN":
        sys.path.append("../OD_VGAN/")
        #cfg["run"]["net_codec"] = torch.load(cfg["general"]["od_vgan_model_path"]).to(cfg["run"]["device"])      
        from codec_od_vgan import codec_ODVGAN 
        cfg["run"]["net_codec"] = codec_ODVGAN(cfg).to(cfg["run"]["device"])
    elif cfg['general']['codec'] == "jpeg_real_ffmpeg":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        os.system("mkdir /dev/shm/sinyukov.m/")
        cfg["run"]["net_codec"] = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 0, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general']['codec'] == "jpeg_real_pil":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        cfg["run"]["net_codec"] = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 1, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general']['codec'] == "jpeg_real_cv2":
        from compress_ffmpeg_jpeg import codec_jpeg_real
        cfg["run"]["net_codec"] = codec_jpeg_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'], mode = 2, x_hat_format = True).to(cfg["run"]["device"])
    elif cfg['general']['codec'] == "img_h264_real":
        from compress_H264 import codec_H264_real
        cfg["run"]["net_codec"] = codec_H264_real(cfg['general']['quality'], compress_path = cfg['general']['home_dir'] , x_hat_format = True).to(cfg["run"]["device"])
    else:
        cfg["run"]["net_codec"] = cheng2020_attn(quality=cfg["general"]["quality"], pretrained = True, metric = cfg["general"]["codec_metric"]).to(cfg["run"]["device"]).requires_grad_(True)# ssf2020 -- video
    
    if cfg["run"]["loss_calc"] == None:
        cfg["run"]["loss_calc"] = Custom_enh_Loss(target_lst = cfg["general"]["met_names"], k_lst = cfg["general"]["k_lst"]).eval().requires_grad_(False).to(cfg["run"]["device"])
        #cfg["run"]["loss_calc"] = Custom_enh_Loss(target_lst = cfg["general"]["met_names"], k_lst = cfg["general"]["k_lst"]).train().requires_grad_(False).to(cfg["run"]["device"])
    rdLoss = RateDistortionLoss()
    
    to_show = True
    if 0:
        torch.nn.parallel.DistributedDataParallel
        cfg["run"]["loss_calc"] = torch.nn.DataParallel(cfg["run"]["loss_calc"])
        cfg["run"]["net_codec"] =  torch.nn.DataParallel(cfg["run"]["net_codec"])
        cfg["run"]["net_enhance"] = torch.nn.DataParallel(cfg["run"]["net_enhance"])
    
    
    try:
        os.mkdir(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))
    except Exception:
        pass   
    dataset_train = CustomImageDataset(cfg["general"]["dataset_dir"],cfg, train= True, datalen = cfg["general"]["datalen_train"], center_crop = False)
    dataset_test = CustomImageDataset(cfg["general"]["dataset_dir"],cfg, train= False, datalen = cfg["general"]["datalen_test"], center_crop = True)
    dataset_train = DataLoader(dataset_train, batch_size = cfg["general"]["batch_size_train"], shuffle = True, num_workers = cfg["general"]["num_workers"])#8#4#8
    dataset_test = DataLoader(dataset_test, batch_size = cfg["general"]["batch_size_test"], shuffle = False, num_workers = cfg["general"]["num_workers"])#8#4#4

    #if 1:
    #    dataset_train = torch.nn.DataParallel(dataset_train)
    #    dataset_test = torch.nn.DataParallel(dataset_test)

    if cfg["general"]["optimize_image"]:
        X = next(iter(dataset_train)).detach().to(cfg["run"]["device"]).requires_grad_()
        Y = X.clone().detach().to(cfg["run"]["device"]).requires_grad_()
        Y.retain_grad()
        X.retain_grad()
    parameters = set(p for p in cfg["run"]["net_enhance"].parameters()) if not cfg["general"]["optimize_image"] else [X]
    if cfg["general"]["optimizer"] == "AdamW":
        cfg["run"]["optimizer"] = optim.AdamW(parameters, **cfg["general"]["optimizer_opt"])
    elif cfg["general"]["optimizer"] == "Adam":
        cfg["run"]["optimizer"] = optim.Adam(parameters, **cfg["general"]["optimizer_opt"])
    elif cfg["general"]["optimizer"] == "SGD":
        cfg["run"]["optimizer"] = optim.SGD(parameters, **cfg["general"]["optimizer_opt"])
    
    cfg["run"]["EarlyStopping"] = EarlyStopping(cfg["general"]["patience"], 'min')
    cfg["run"]["SaveBestHandler"] = SaveBestHandler(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))

    
    cfg["run"]["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(cfg["run"]["optimizer"], len(dataset_train) * (cfg["general"]["max_epoch"]-cfg["general"]["lr_linear_stage"]), eta_min = cfg["general"]["eta_min"])
    
    save_result = True
    X_sample = torch.load("sample_data/X.ckpt").to("cpu")
    n = 30
    logs_plot_cur = {}
    logs_plot = {}
    skip_0epoch = True
    tmp = []
    if cfg["general"]["use_wandb"]:
        wandb.watch(cfg["run"]["net_enhance"], log='all')
        #wandb.watch(net_codec, log='all')
        wandb.watch(cfg["run"]["loss_calc"], log='all')
    logs_plot_min = 1000000
    
    def calculate_met(times = 1):
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
    
    cfg["run"]["logger"] = Logger(cfg)
    cfg["run"]["logger"].write_cfg()
    ckpt_save_lst = ["net_enhance",]#["optimizer", "net_enhance",]
    
    if cfg["general"]["ckpt_recovery"]:
        print("!CKPT RECOVERY!")
        try:
            if cfg["general"]["ckpt_recovery_path"] == "":
                cfg["general"]["ckpt_recovery_path"] = os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "ckpt.ckpt")
            ckpt_ckpt = torch.load(cfg["general"]["ckpt_recovery_path"])
            for k_i in ckpt_save_lst:
                cfg["run"][k_i].load_state_dict(ckpt_ckpt[k_i])
                print(f"{k_i} loaded from ckpt") 
        except Exception:
            print("CKPT LOAD FAILED")
            raise
        if cfg["general"]["ckpt_recovery_path"] == "" or cfg["general"]["ckpt_recovery_path"] == os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "ckpt.ckpt"):
            try:
                import pickle
                with open(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "plots.pkl"),'rb') as f:
                    logs_plot = pickle.load(f)
                del ckpt_ckpt
            except Exception:
                print("LOG LOAD FAILED")
                if cfg["general"]["ckpt_recovery_path"] == "" or cfg["general"]["ckpt_recovery_path"] == os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "ckpt.ckpt"):
                    raise
            
            
        
    for epoch in tqdm(range(cfg["general"]["max_epoch"])):
        if cfg["general"]["break_flag"] == True:
            break
        idx_video = 0
        logs_plot_cur = {}
        if skip_0epoch and epoch == 0:
            continue
        gradnorm_max = 0
        for to_train in [True, False]:
            tqdm_dataset = tqdm(dataset_train if to_train else dataset_test)
            cfg["run"]["net_enhance"].train(to_train)
            with (torch.enable_grad() if (to_train and cfg["general"]["train_mode"]) else torch.no_grad()):
                for frame in tqdm_dataset:
                    idx_video += 1
                    if not cfg["general"]["optimize_image"]:
                        X = frame.to(cfg["run"]["device"]).requires_grad_()
                        X.retain_grad()
                        #X = torchvision.transforms.RandomResizedCrop((patch_sz,patch_sz))(X)
                        Y = X.clone().detach().to(cfg["run"]["device"]).requires_grad_(False)
                        #Y.retain_grad()
                    #X.data.clamp_(min=0,max=1)
                    cfg["run"]["optimizer"].zero_grad()
                    #aux_optimizer.zero_grad()
                    if not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0:
                        X_enhance = cfg["run"]["net_enhance"](X)
                        if cfg["general"]["enhance_net"] == "GAPresnet" or \
                            cfg["general"]["enhance_net"] == "GAPunet":
                            X_enhance = X_enhance + X
                        elif cfg["general"]["sigmoid_activation"] == 0:#Fails for default
                            pass
                        elif cfg["general"]["sigmoid_activation"] == 1: #Fine for default, bit better than 2
                            X_enhance = torch.sigmoid(X_enhance)
                        elif cfg["general"]["sigmoid_activation"] == 2:#Fine for default
                            X_enhance = torch.tanh(X_enhance) + X
                        elif cfg["general"]["sigmoid_activation"] == 3:#Fails for default
                            X_enhance = X_enhance + X
                        X_enhance.data.clamp_(min=0,max=1)
                        #X_enhance = torch.maximum(X_enhance, torch.zeros_like(X_enhance).to(cfg["run"]["device"]))
                        #X_enhance = torch.minimum(X_enhance, torch.ones_like(X_enhance).to(cfg["run"]["device"]))
                        X_out = cfg["run"]["net_codec"].forward(X_enhance)
                        
                        X_out['x_hat'].data.clamp_(min=0,max=1)
                        X_out["x_hat"] = X_out["x_hat"][..., :X_enhance.shape[-2], :X_enhance.shape[-1]]
                    else:
                        X_codec = cfg["run"]["net_codec"].forward(X)
                        X_codec['x_hat'].data.clamp_(min=0,max=1)
                        X_codec["x_hat"] = X_codec["x_hat"][..., :X_codec['x_hat'].shape[-2], :X_codec['x_hat'].shape[-1]]
                        X_out = {"likelihoods" : X_codec["likelihoods"]}
                        X_codec['x_hat'].data.clamp_(min=0,max=1)
                        X_out["x_hat"] = cfg["run"]["net_enhance"](X_codec["x_hat"])
                        #X_enhance = torch.sigmoid(X_enhance)
                        if cfg["general"]["sigmoid_activation"] == 0:
                            pass
                        elif cfg["general"]["sigmoid_activation"] == 1:            
                            X_out["x_hat"] = torch.sigmoid(X_out["x_hat"])
                        elif cfg["general"]["sigmoid_activation"] == 2:
                            X_out["x_hat"] = torch.tanh(X_out["x_hat"]) + X_codec["x_hat"]
                        elif cfg["general"]["sigmoid_activation"] == 3:
                            X_out["x_hat"] = X_out["x_hat"] + X_codec["x_hat"]
                        
                        if "bpp_loss" in X_codec.keys():
                            X_out['bpp_loss'] = X_codec['bpp_loss']
                        
                    #X_out['x_hat'] = torch.nan_to_num(X_out['x_hat'])
                    #Y = torch.nan_to_num(Y)
                    
                    if str(X.mean().item()) == 'nan' or str(X_out['x_hat'].mean().item()) == 'nan':
                        continue
                    loss = cfg["run"]["loss_calc"](X_out, Y)
                    if str(loss[list(loss.keys())[4]].item()) == 'nan':
                        print("Exception: NAN in loss")
                    lmbda = 1e-2
        
                    if epoch != 0 and to_train and cfg["general"]["train_mode"]:
                        loss["loss"].backward()
                        for p in list(filter(lambda p: p.grad is not None, parameters)):
                            #gradnorm_cur = abs(p.grad.data.norm(2).item())
                            gradnorm_cur = abs(p.grad.data.norm(float('inf')).item())
                            if gradnorm_max <= gradnorm_cur:
                                print(gradnorm_max)
                                gradnorm_max = gradnorm_cur
                        #if gradnorm_cur > 1.:
                        #    continue
                        torch.nn.utils.clip_grad_norm_(parameters, 1.)
                        #torch.nn.utils.clip_grad_norm_(parameters, 10.)#0.05#0.1
                        #torch.nn.utils.clip_grad_value_(parameters, 1.)
                        #for par in parameters:
                            #if par.grad != None:
                                #par.grad = torch.nan_to_num(par.grad)
                        #list(parameters)[0] = torch.nan_to_num(list(parameters)[0])
                        if (1 or gradnorm_cur < 0.025) and cfg["general"]["train_mode"]:
                            cfg["run"]["optimizer"].step()
                        if epoch > cfg["general"]["lr_linear_stage"] and cfg["general"]["train_mode"]:
                            cfg["run"]["scheduler"].step()
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
                
        print("gradient norm: {}".format(gradnorm_max))
        if cfg["general"]["use_wandb"]:
            wandb.log({"gradient norm max": gradnorm_max}, step = epoch)
        if not to_train:
            for j in list(logs_plot_cur.keys()):
                if not j in logs_plot:
                    logs_plot[j] = []
                logs_plot[j].append(np.mean(logs_plot_cur[j]))
                if cfg["general"]["use_wandb"]:
                    wandb.log({j: np.mean(logs_plot_cur[j])}, step = epoch)
        if cfg["general"]["save_ckpt"]:
            torch.save({k_i : cfg["run"][k_i].state_dict() for k_i in ckpt_save_lst},  os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "ckpt.ckpt"))
        clear_output()
        fig = plt.figure(figsize=(20,8))
        unique_names = list(np.unique(list(map(lambda x : x.split("_test")[0], list(logs_plot.keys())))))
        unique_idx = {i: j for i, j in zip(unique_names, range(len(unique_names)))}
        for plot_idx, plot_i in enumerate(list(logs_plot.keys())):
            short_name = plot_i.split("_test")[0]
            cur_idx = unique_idx[short_name]
            plt.subplot(2, math.ceil( (len(unique_idx)+1) / 2), cur_idx + 1)
            plt.xlabel("step_number")
            plt.title("Learning curve " + short_name)
            plt.ylabel(short_name)
            plt.plot(np.arange(len(logs_plot[plot_i])), logs_plot[plot_i], label = ("train" if short_name == plot_i else "test"))
            plt.legend()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            plt.tight_layout()
        if save_result == True:
            fig.savefig(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "lerningcurve.png"))
            if cfg["general"]["save_netcodec"] == True:
                torch.save(cfg["run"]["net_codec"].state_dict(), os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "model_codec.ckpt"))
            if cfg["general"]["save_net_enhance"] and cfg["run"]["net_enhance"] != None and cfg["run"]["net_enhance"] != enhance_Identity:
                cfg["run"]["SaveBestHandler"](cfg["run"]["net_enhance"], epoch, logs_plot["loss_test"][-1], optimizer = cfg["run"]["optimizer"], scheduler = cfg["run"]["scheduler"])
                #if len(logs_plot["loss_test"]) < 2 or logs_plot_min > logs_plot["loss_test"][-1]:
                    #logs_plot_min = logs_plot["loss_test"][-1]
                    #torch.save(cfg["run"]["net_enhance"].state_dict(), os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "model_net.ckpt"))
            import pickle
            with open(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "plots.pkl"), 'wb') as f:
                pickle.dump(logs_plot, f)
        
        tqdm_dataset.refresh()
        if to_show:
            plt.show()
        plt.figure(25)
        #X.data = X_sample.data
        #X_out = net_codec.forward(X)
        pltimshow_batch([Y, (X_enhance if (not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0) else X_codec['x_hat']) , X_out['x_hat']], filename = os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "vis.png"))
        if cfg["general"]["use_wandb"]:
            wandb.log({"Enhanced": wandb.Image((X_enhance if (not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0) else X_codec['x_hat'])), "Enhanced + Compressed": wandb.Image(X_out['x_hat']),  "GT": wandb.Image(Y)}, step = epoch)
        if to_show:
            plt.pause(0.005)
            
        if cfg["run"]["EarlyStopping"](logs_plot["loss_test"][-1]):
            break
    
