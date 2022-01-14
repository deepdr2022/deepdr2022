import pickle
import numpy as np
# from torchinfo import summary
import torch 
from dataLoader import make_dataloader
import model
import yaml
import pandas as pd
import time
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
import argparse


def test(model, loaders,model_name, need_mask=True):
    res = {}
    for frac in ["30", "40", "50", "60", "70", "80"]:
        # print(frac)
        res[frac + "rmse"] = []
        res[frac + "rmspe"] = []
        res[frac + "mae"] = []
        res[frac + "time"] = []

        for clean, masked in loaders["test"+frac]:
            if model_name == "CNN":
                masked = torch.unsqueeze(masked, 1).type(torch.FloatTensor)
            else:
                masked = masked.view(masked.size(0),-1).type(torch.FloatTensor)
            mask = torch.isnan(masked)
            masked = torch.nan_to_num(masked, nan=0.0)
            masked = masked.to(device)
            mask = mask.to(device)
            if model_name == "CNN":
                clean = torch.unsqueeze(clean, 1).type(torch.FloatTensor)
            else:
                clean = clean.view(clean.size(0),-1).type(torch.FloatTensor)
            # Move tensor to the proper device
            if need_mask:
                start = time.time()
                completed = model(masked, mask)
                t = time.time() - start
            else:
                start = time.time()
                completed = model(masked)
                t = time.time() - start
            completed = torch.clamp(completed, min=0, max=1)
            completed = completed.cpu().detach().numpy()
            clean = clean.numpy()
            mask = mask.cpu().numpy()
            distance = completed[mask] - clean[mask]

            rmse = np.sqrt((distance ** 2).mean())

                
            division = np.divide(distance, clean[mask], out=np.zeros_like(distance), where=clean[mask]>0) 
            rmspe = np.sqrt((division ** 2).mean())
            mae = np.abs(distance).mean()
            res[frac + "rmse"].append(rmse)
            res[frac + "rmspe"].append(rmspe)
            res[frac + "mae"].append(mae)
            res[frac + "time"].append(t)

    df = pd.DataFrame(res)
    df = df.replace(np.inf, np.nan)
    df = df.append(df.describe())
    return df
    
def load_model(model_path, dropout, dataset, model_name,bs=1):
    if model_name == "DenoisingAutoEncoder":
        conf_file = "conf.yml"
        with open(conf_file, "r") as f:
            conf = yaml.safe_load(f)
        dims=conf["dims"][dataset]
        models = model.DenoisingAutoEncoder(dims=dims, dropout_ratio=dropout)
        models.load_state_dict(torch.load(model_path))
    elif model_name == "DAEM":        
        conf_file = "conf_m.yml"
        with open(conf_file, "r") as f:
            conf = yaml.safe_load(f)
        dims=conf["dims"][dataset]
        models = model.DAEM(dims=dims, dropout_ratio=dropout)
        models.load_state_dict(torch.load(model_path))
    elif model_name == "CNN":
        conf_file = "conf_cnn.yml"
        models = model.CNN(dataname=dataset)
        models.load_state_dict(torch.load(model_path))
    elif model_name == "DeepDR":
        conf_file = "conf_deepdr.yml"
        with open(conf_file, "r") as f:
            conf = yaml.safe_load(f)
        feature_in=conf["feature_in"][dataset]
        feature_hid=conf["feature_hid"][dataset]
        heads=conf["heads"][dataset]
        num_layers=conf["num_layers"][dataset]
        models = model.AttentionWithPE(feature_in=feature_in, feature_hid=feature_hid*heads, heads=heads, num_layers=num_layers)
        models.load_state_dict(torch.load(model_path)['model_state_dict'])
    else:
        print("Model not supported!")
        exit(1)

    loaders = make_dataloader(dataset, batch_size=bs)
    models.to(device)
    models.eval()     
    return models, loaders
def run_test(model_name, dataset, model_path,dr=0.0):
    mod, loader = load_model(model_path, dataset=dataset,  model_name=model_name, dropout=dr)
    need_mask = True
    if model_name == "DenoisingAutoEncoder":
        need_mask = False
    return test(mod, loader, model_name, need_mask=need_mask)

    
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--modelpath', type=str)
args = parser.parse_args()
dataname = args.dataname
modelpath = args.modelpath
mod = args.model

df = run_test(mod, dataname, modelpath)
print(df.tail(10))
df.to_csv(dataname + '_' + mod + '.csv')