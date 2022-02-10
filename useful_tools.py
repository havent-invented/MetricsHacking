from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
def count_params_all_3():
    print("net_enhance")
    count_parameters(net_enhance)
    print("net_codec")
    count_parameters(net_codec)
    print("metr.model")
    count_parameters(metr.model)
    print("net_enhance.base_model")
    count_parameters(net_enhance.base_model)
