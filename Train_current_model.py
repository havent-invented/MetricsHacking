import sys

if cfg["general"]["use_wandb"]:
    import wandb
    wandb.init(project=cfg["general"]["project_name"], entity="havent_invented", name = cfg["general"]["name"], tags = {"Train"}, save_code = True)
    #wandb.config.update({"general": cfg["general"]})
sys.path.insert(1, "E:/VMAF_METRIX/NeuralNetworkCompression/")
exec(open('main.py').read())#MAIN

X = None
cfg["run"]["loss_calc"] = Custom_enh_Loss(target_lst =cfg["general"]["met_names"], k_lst = cfg ["general"] ["k_lst"]).eval().requires_grad_(True).to(cfg["run"]["device"])

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
            
            
if cfg["general"]["codec"] == "No":
    cfg["run"]["net_codec"] = codec_Identity
else:
    cfg["run"]["net_codec"] = cheng2020_attn(quality=cfg["general"]["quality"], pretrained=True).to(cfg["run"]["device"]).requires_grad_(True)# ssf2020 -- video

#env = calc_met( model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir)
#self = env
if cfg["run"]["loss_calc"] == None:
    cfg["run"]["loss_calc"] = Custom_enh_Loss(target_lst = cfg["general"]["met_names"], k_lst = cfg["general"]["k_lst"]).eval().requires_grad_(True).to(cfg["run"]["device"])
rdLoss = RateDistortionLoss()

try:
    os.mkdir(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))
except Exception:
    pass   
dataset_train = CustomImageDataset(dst_dir_vimeo,train= True, datalen = cfg["general"]["datalen_train"], center_crop = False)
dataset_test = CustomImageDataset(dst_dir_vimeo,train= False, datalen = cfg["general"]["datalen_test"], center_crop = True)
dataset_train = DataLoader(dataset_train, batch_size = cfg["general"]["batch_size_train"], shuffle = True)#8#4#8
dataset_test = DataLoader(dataset_test, batch_size = cfg["general"]["batch_size_test"], shuffle = False)#8#4#4
mse_loss = nn.MSELoss()
#opt_target = [i for i in net_codec.parameters()]
#opt_target = [p for n,p in net_codec.named_parameters()]
#optimizer = optim.Adam(opt_target, lr = 0.001)
AV_log = []
curve_mse = []
plot_data = []
plot_data_mse = []

if cfg["general"]["optimize_image"]:
    X = next(iter(dataset_train)).detach().to(cfg["run"]["device"]).requires_grad_()
    Y = X.clone().detach().to(cfg["run"]["device"]).requires_grad_()
    Y.retain_grad()
    X.retain_grad()
parameters = set(p for p in cfg["run"]["net_enhance"].parameters()) if not cfg["general"]["optimize_image"] else [X]
#aux_parameters = set(p for n, p in net_codec.named_parameters() if n.endswith(".quantiles"))
#aux_loss = net_codec.entropy_bottleneck.loss()
if cfg["general"]["optimizer"] == "AdamW":
    cfg["run"]["optimizer"] = optim.AdamW(parameters, **cfg["general"]["optimizer_opt"])
if cfg["general"]["optimizer"] == "Adam":
    cfg["run"]["optimizer"] = optim.Adam(parameters, **cfg["general"]["optimizer_opt"])
elif cfg["general"]["optimizer"] == "SGD":
    cfg["run"]["optimizer"] = optim.SGD(parameters, **cfg["general"]["optimizer_opt"])

cfg["run"]["EarlyStopping"] = EarlyStopping(cfg["general"]["patience"], 'min')
cfg["run"]["SaveBestHandler"] = SaveBestHandler(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))

#aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)


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
        for frame in tqdm_dataset:
            idx_video += 1
            if not cfg["general"]["optimize_image"]:
                X = frame.to(cfg["run"]["device"]).requires_grad_()
                X.retain_grad()
                #X = torchvision.transforms.RandomResizedCrop((patch_sz,patch_sz))(X)
                Y = X.clone().detach().to(cfg["run"]["device"]).requires_grad_()
                Y.retain_grad()
            #X.data.clamp_(min=0,max=1)
            cfg["run"]["optimizer"].zero_grad()
            #aux_optimizer.zero_grad()
            X_enhance = cfg["run"]["net_enhance"](X)
            #X_enhance = torch.sigmoid(X_enhance)
            X_enhance.data.clamp_(min=0,max=1)
            X_out = cfg["run"]["net_codec"].forward(X_enhance)
            X_out['x_hat'].data.clamp_(min=0,max=1)
            X_out["x_hat"] = X_out["x_hat"][..., :X_enhance.shape[-2], :X_enhance.shape[-1]]
            #X_out['x_hat'] = torch.nan_to_num(X_out['x_hat'])
            #Y = torch.nan_to_num(Y)
            
            
            loss = cfg["run"]["loss_calc"](X_out, Y)
            if str(loss[list(loss.keys())[4]].item()) == 'nan':
                print("Exception: NAN in loss")
            lmbda = 1e-2

            if epoch != 0 and to_train:
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
                torch.nn.utils.clip_grad_value_(parameters, 1.)
                for par in parameters:
                    if par.grad != None:
                        par.grad = torch.nan_to_num(par.grad)
                #list(parameters)[0] = torch.nan_to_num(list(parameters)[0])
                if gradnorm_cur < 0.025:
                    cfg["run"]["optimizer"].step()
                if epoch > cfg["general"]["lr_linear_stage"]:
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
        wandb.log({"gradient norm max": gradnorm_max})
    if not to_train:
        for j in list(logs_plot_cur.keys()):
            if not j in logs_plot:
                logs_plot[j] = []
            logs_plot[j].append(np.mean(logs_plot_cur[j]))
            if cfg["general"]["use_wandb"]:
                wandb.log({j: np.mean(logs_plot_cur[j])})
    if 1:
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
        plt.show()
        plt.figure(25)
        #X.data = X_sample.data
        #X_out = net_codec.forward(X)
        pltimshow_batch([Y, X_enhance, X_out['x_hat']], filename = os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "vis.png"))
        if cfg["general"]["use_wandb"]:
            wandb.log({"Enhanced": wandb.Image(X_enhance), "Enhanced + Compressed": wandb.Image(X_out['x_hat']),  "GT": wandb.Image(Y)})
        plt.pause(0.005)
        
    if cfg["run"]["EarlyStopping"](logs_plot["loss_test"][-1]):
        break