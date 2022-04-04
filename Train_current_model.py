import sys

try:
    use_wandb
except Exception:
    use_wandb = True
if use_wandb:
    import wandb
    wandb.init(project="White box training", entity="havent_invented", name="Tests White box full-reference training", tags = {"Train"}, save_code = True)
    try:
        config_dictionary
    except Exception:
        config_dictionary = {}
    wandb.config.update(config_dictionary)
sys.path.insert(1, "E:/VMAF_METRIX/NeuralNetworkCompression/")
exec(open('main.py').read())#MAIN
try:
    patch_sz
except Exception:
    patch_sz = 256
try:
    save_filename
except Exception:
    save_filename = "tmp"
try: 
    break_flag
except Exception:
    break_flag = False
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
    
if net_enhance == None:
    net_enhance = ResNetUNet(3).to(device)
    #net_enhance = nn.Sequential(nn.Conv2d(3, 3, 3,1, "same"),)
    net_enhance = net_enhance.to(device)
    if not break_flag:
        net_enhance = net_enhance.requires_grad_(True)
        for i in net_enhance.parameters():
            i.retain_grad()
if net_codec == None:
    net_codec = cheng2020_attn(quality=args_p.quality, pretrained=True).to(device).requires_grad_(True)# ssf2020 -- video
    #.eval()
    #if net_codec != codec_Identity:
        #net_codec = net_codec.requires_grad_(True)#####Fix eval

    
    
env = calc_met( model = "MDTVSFA", home_dir1=home_dir,dataset_dir=dst_dir)
self = env
if loss_calc == None:
    loss_calc = Custom_enh_Loss()
rdLoss = RateDistortionLoss()
try:
    batch_size_train
except Exception:
    batch_size_train = 8
try:
    batch_size_test
except Exception:
    batch_size_test = 2
    
dataset_train = CustomImageDataset(dst_dir_vimeo,train= True, datalen = datalen_train, center_crop = False)
dataset_test = CustomImageDataset(dst_dir_vimeo,train= False, datalen = datalen_test, center_crop = True)
dataset_train = DataLoader(dataset_train, batch_size = batch_size_train, shuffle = True)#8#4#8
dataset_test = DataLoader(dataset_test, batch_size = batch_size_test, shuffle = True)#8#4#4
mse_loss = nn.MSELoss()
#opt_target = [i for i in net_codec.parameters()]
#opt_target = [p for n,p in net_codec.named_parameters()]
#optimizer = optim.Adam(opt_target, lr = 0.001)
AV_log = []
curve_mse = []
plot_data = []
plot_data_mse = []

if optimize_image:
    X = next(iter(dataset_train)).detach().to(device).requires_grad_()
    Y = X.clone().detach().to(device).requires_grad_()
    Y.retain_grad()
    X.retain_grad()
parameters = set(p for p in net_enhance.parameters()) if not optimize_image else [X]
#aux_parameters = set(p for n, p in net_codec.named_parameters() if n.endswith(".quantiles"))
#aux_loss = net_codec.entropy_bottleneck.loss()

#optimizer = optim.AdamW(parameters, lr=0.0001, weight_decay = 0.01)#
#optimizer = optim.Adam(parameters, lr=1e-4)
optimizer = optim.SGD(parameters, lr = 0.001)
#aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

save_result = True
X_sample = torch.load("sample_data/X.ckpt").to("cpu")
n = 30
logs_plot_cur = {}
logs_plot = {}
try:
    max_epoch
except Exception:
    max_epoch = 12
skip_0epoch = True
tmp = []
wandb.watch(net_enhance, log='all')
#wandb.watch(net_codec, log='all')
wandb.watch(loss_calc, log='all')
logs_plot_min = 1000000
break_flag = True
def calculate_met(times = 1):
    for to_train in [True, False]:
        tqdm_dataset = tqdm(dataset_train if to_train else dataset_test)
        for frame in tqdm_dataset:
            if not optimize_image:
                X = frame.to(device)
                Y = X.clone().detach().to(device)
            X_enhance = enhance_Identity(X)
            X_enhance.data.clamp_(min=0,max=1)
            X_out = net_codec.forward(X_enhance)
            X_out['x_hat'].data.clamp_(min=0,max=1)
            loss = loss_calc(X_out, Y)
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

for epoch in tqdm(range(max_epoch)):
    if break_flag == True:
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
            if not optimize_image:
                X = frame.to(device).requires_grad_()
                X.retain_grad()
                #X = torchvision.transforms.RandomResizedCrop((patch_sz,patch_sz))(X)
                Y = X.clone().detach().to(device).requires_grad_()
                Y.retain_grad()
            #X.data.clamp_(min=0,max=1)
            optimizer.zero_grad()
            #aux_optimizer.zero_grad()
            X_enhance = net_enhance(X)
            #X_enhance = torch.sigmoid(X_enhance)
            X_enhance.data.clamp_(min=0,max=1)
            X_out = net_codec.forward(X_enhance)
            X_out['x_hat'].data.clamp_(min=0,max=1)
            
            #X_out['x_hat'] = torch.nan_to_num(X_out['x_hat'])
            #Y = torch.nan_to_num(Y)
            
            loss = loss_calc(X_out, Y)
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

    print("gradient norm: {}".format(gradnorm_max))
    wandb.log({"gradient norm max": gradnorm_max})
    if not to_train:
        for j in list(logs_plot_cur.keys()):
            if not j in logs_plot:
                logs_plot[j] = []
            logs_plot[j].append(np.mean(logs_plot_cur[j]))
            if use_wandb:
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
            fig.savefig("vis/lerningcurve" + save_filename + ".png")
            if save_netcodec == True:
                torch.save(net_codec.state_dict(), "models/model_" +save_filename + ".ckpt") 
            if save_net_enhance and net_enhance != None and net_enhance != enhance_Identity:
                if len(logs_plot["loss_test"]) < 2 and logs_plot_min > logs_plot["loss_test"][-1]:
                    logs_plot_min = logs_plot["loss_test"][-1]
                    torch.save(net_enhance.state_dict(), "models_enhancement_FR/model_" +save_filename + ".ckpt") 
            import pickle
            with open('logs_enhancement/plots'+ save_filename + '.pkl', 'wb') as f:
                pickle.dump(logs_plot, f)
    
        tqdm_dataset.refresh()
        plt.show()
        plt.figure(25)
        #X.data = X_sample.data
        #X_out = net_codec.forward(X)
        pltimshow_batch([Y, X_enhance, X_out['x_hat']], filename = "vis/pics_" + save_filename + ".png")
        if use_wandb:
            wandb.log({"Enhanced": wandb.Image(X_enhance), "Enhanced + Compressed": wandb.Image(X_out['x_hat']),  "GT": wandb.Image(Y)})
        plt.pause(0.005)
