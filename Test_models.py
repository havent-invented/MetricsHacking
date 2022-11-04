import sys
from Current_model_lib import *
import ntpath
from components import *

def test(cfg):
    if cfg["general"]["use_wandb"]:
        import wandb
        wandb.init(project=cfg["general"]["project_name"], entity="havent_invented", name = cfg["general"]["name"], tags = {"Train"}, save_code = True)
        wandb.config.update({"general": cfg["general"]})
    
    if not os.path.exists(cfg["general"]["home_dir"]):
        os.makedirs(cfg["general"]["home_dir"])
    X = None
    cfg["run"]["loss_calc"] = Custom_enh_Loss_seq(target_lst = cfg["general"]["met_names"], k_lst = cfg ["general"] ["k_lst"]).eval().requires_grad_(True).to(cfg["run"]["device"])
    
    get_enhance_net(cfg)
    #get_codec(cfg)

    get_codec(cfg, src_key = 'codec', target_key = 'net_codec')

    if "pretrained_model_path" in cfg["general"]:
        ckpt = torch.load(cfg["general"]['pretrained_model_path'])['model']
        cfg["run"]["net_enhance"].load_state_dict(ckpt)
    try:
        os.mkdir(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"]))
    except Exception:
        pass   
    dataset= CustomImageDataset(cfg["general"]["dataset_dir"],cfg, train = False, datalen = cfg["general"]["datalen_train"], center_crop = True, return_name = True)
    dataset= DataLoader(dataset, shuffle = False, num_workers = cfg["general"]["num_workers"], batch_size = 1)#8#4#8
    #dataset_train = CustomImageDataset(cfg["general"]["dataset_dir"],cfg, train= True, datalen = cfg["general"]["datalen_train"], center_crop = False, return_name = True)
    #dataset_test = CustomImageDataset(cfg["general"]["dataset_dir"],cfg, train= False, datalen = cfg["general"]["datalen_test"], center_crop = True, return_name = True)
    #dataset_train = DataLoader(dataset_train, batch_size = cfg["general"]["batch_size_train"], shuffle = True, num_workers = cfg["general"]["num_workers"])#8#4#8
    #dataset_test = DataLoader(dataset_test, batch_size = cfg["general"]["batch_size_test"], shuffle = False, num_workers = cfg["general"]["num_workers"])#8#4#4
    #parameters = set(p for p in cfg["run"]["net_enhance"].parameters())
    #get_optim(cfg, parameters)
    
    save_result = True
    logs_plot_cur = {}
    logs_plot = {}
    
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

    
    
    idx_video = 0
    logs_plot_cur = {}
    gradnorm_max = 0
    stats_log = {}
    tqdm_dataset = tqdm(dataset)
    cfg["run"]["net_enhance"].train(False)
    with torch.no_grad():
        for frame, frame_nams in tqdm_dataset:
            idx_video += 1
            if not cfg["general"]["optimize_image"]:
                X = frame.to(cfg["run"]["device"]).requires_grad_()
                X.retain_grad()
                Y = X.clone().detach().to(cfg["run"]["device"]).requires_grad_()
                Y.retain_grad()
            if not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0:
                X_enhance = cfg["run"]["net_enhance"](X)
                if cfg["general"]["sigmoid_activation"] == True: 
                    X_enhance = torch.sigmoid(X_enhance)
                X_enhance.data.clamp_(min=0,max=1)
                X_out = cfg["run"]["net_codec"].forward(X_enhance)
                X_out['x_hat'].data.clamp_(min=0,max=1)
                X_out["x_hat"] = X_out["x_hat"][..., :X_enhance.shape[-2], :X_enhance.shape[-1]]
                if cfg["general"]["save_enh"]:
                    for fr_idx in range(len(frame_nams)):
                        import ntpath
                        out_img_nam = ntpath.basename(frame_nams[fr_idx])
                        cfg["run"]["logger"].save_img_from_dataset(X_enhance[fr_idx], out_img_nam, subdir = "enh")
            else:
                X_codec = cfg["run"]["net_codec"].forward(X)
                X_codec['x_hat'].data.clamp_(min=0,max=1)
                X_codec["x_hat"] = X_codec["x_hat"][..., :X_codec['x_hat'].shape[-2], :X_codec['x_hat'].shape[-1]]
                X_out = {"likelihoods" : X_codec["likelihoods"]}
                X_codec['x_hat'].data.clamp_(min=0,max=1)
                X_out["x_hat"] = cfg["run"]["net_enhance"](X_codec["x_hat"])
                if cfg["general"]["sigmoid_activation"] == True:            
                    X_out["x_hat"] = torch.sigmoid(X_out["x_hat"])
                if "bpp_loss" in X_codec.keys():
                    X_out['bpp_loss'] = X_codec['bpp_loss']
            if cfg["general"]["save_out"]:
                for fr_idx in range(len(frame_nams)):
                    import ntpath
                    #out_img_nam = os.path.splitext(ntpath.basename(frame_nam[fr_idx]))[0]
                    out_img_nam = ntpath.basename(frame_nams[fr_idx])
                    cfg["run"]["logger"].save_img_from_dataset(X_out["x_hat"][fr_idx], out_img_nam, subdir = "out")
            
            if str(X.mean().item()) == 'nan' or str(X_out['x_hat'].mean().item()) == 'nan':
                continue
            loss = cfg["run"]["loss_calc"](X_out, Y)
            import ntpath
            for loss_key in loss.keys():
                for fr_nam in frame_nams:
                    out_img_nam = ntpath.basename(fr_nam)
                    if not out_img_nam in stats_log:
                        stats_log[out_img_nam] = {}
                    stats_log[out_img_nam][loss_key] = loss[loss_key].item()

            if str(loss[list(loss.keys())[4]].item()) == 'nan':
                print("Exception: NAN in loss")                   
            for j in list(loss.keys()):
                j_converted = j 
                if not j_converted in logs_plot_cur:
                    logs_plot_cur[j_converted] = []
                logs_plot_cur[j_converted].append(loss[j].data.to("cpu").numpy())
        
    print("gradient norm: {}".format(gradnorm_max))
    if cfg["general"]["use_wandb"]:
        wandb.log({"gradient norm max": gradnorm_max}, step = 0)
    
    for j in list(logs_plot_cur.keys()):
        if not j in logs_plot:
            logs_plot[j] = []
        logs_plot[j].append(np.mean(logs_plot_cur[j]))
        if cfg["general"]["use_wandb"]:
            wandb.log({j: np.mean(logs_plot_cur[j])}, step = 0)
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
        plt.scatter(np.arange(len(logs_plot[plot_i])), logs_plot[plot_i])
        plt.legend()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.tight_layout()
    if save_result == True:
        fig.savefig(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "lerningcurve.png"))
        import pickle
        with open(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "plots.pkl"), 'wb') as f:
            pickle.dump(logs_plot, f)
        with open(os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "stats.pkl"), 'wb') as f:
            pickle.dump(stats_log, f)
    tqdm_dataset.refresh()
    plt.show()
    plt.figure(25)
    pltimshow_batch([Y, (X_enhance if (not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0) else X_codec['x_hat']) , X_out['x_hat']], filename = os.path.join(cfg["general"]["logs_dir"], cfg["general"]["name"], "vis.png"))
    if cfg["general"]["use_wandb"]:
        wandb.log({"Enhanced": wandb.Image((X_enhance if (not "order_pre_post" in cfg['general'] or cfg['general']['order_pre_post'] == 0) else X_codec['x_hat'])), "Enhanced + Compressed": wandb.Image(X_out['x_hat']),  "GT": wandb.Image(Y)}, step = 0)
    plt.pause(0.005)

