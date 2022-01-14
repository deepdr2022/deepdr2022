import numpy as np # this module is useful to work with numerical arrays
import random 
import torch
import os
from dataLoader import make_dataloader

from model import AttentionWithPE
import yaml
#import ray
import pandas as pd
# reproducibility
torch.manual_seed(5)
np.random.seed(5)
random.seed(5)

from torch.utils.tensorboard import SummaryWriter



def mask_data(input, mask_frac, mask_to=0.0):
    rand_mask = torch.rand(input.size(),dtype=input.dtype,layout=input.layout, device=input.device) < mask_frac

    return input * ~rand_mask + mask_to * rand_mask, rand_mask

def train_epoch(G, opt_g,device, dataloader, mf):
    G.train()

    g_losses = []
    for idx, (data_batch, _) in enumerate(dataloader): # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        data_batch = data_batch.view(data_batch.size(0),-1).type(torch.FloatTensor)

        masked_data, mask = mask_data(data_batch, mask_frac=mf)
        data_batch, masked_data, mask = data_batch.to(device), masked_data.to(device), mask.to(device)

        output = G(masked_data, mask)
        filled_out, filled_batch = output * mask, data_batch * mask

        g_loss = torch.sqrt(torch.nn.MSELoss()(filled_out, filled_batch))

        opt_g.zero_grad()
        g_loss.backward(inputs=list(G.parameters()))
        opt_g.step()
        g_losses.append(g_loss.item())

    return np.mean(g_losses)

def val_epoch(G, device, dataloader):
    G.eval()
    with torch.no_grad():
        res_all = []
        for clean, masked in dataloader:
            masked = masked.view(masked.size(0),-1).type(torch.FloatTensor)
            mask = torch.isnan(masked)
            masked = torch.nan_to_num(masked, nan=0.0)
            clean = clean.view(clean.size(0),-1).type(torch.FloatTensor)
            masked, mask = masked.to(device), mask.to(device)

            output = G(masked, mask)

            clean = clean.numpy()
            mask = mask.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            filled_out, filled_clean = output[mask], clean[mask]
            res_all.append(np.sqrt(((filled_out - filled_clean)**2).mean()))
    return np.array(res_all).mean()


def train(config,  writer, checkpoint_dir=None):
    bs = config["bs"]
    nl = config["layers"]
    hd = config["heads"]
    fh = config["hid"]
    mf = config["mask_frac"]
    lr = config["lr"]
    feature_in = config["feature_in"]
    ac = config["ac"]
    num_epochs = config["num_epochs"]

    loaders = make_dataloader(dataname, batch_size=bs, num_workers=8)
    G = AttentionWithPE(feature_in=feature_in, feature_hid=fh*hd, heads=hd, num_layers=nl)
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(5)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    G.to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr)

    epoches = 0
    if checkpoint_dir:
        states = torch.load(
            checkpoint_dir)
        G.load_state_dict(states["model_state_dict"])
        opt_g.load_state_dict(states["optim_state_dict"])
        epoches = states["epoch"]
    global_min = 100
    for epoch in range(epoches, num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        g_loss=train_epoch(
            G = G,
            opt_g=opt_g,
            device=device, 
            dataloader=loaders["train"],
            mf=mf
        ) 
        print(f"{mf}_train loss: {g_loss}")
        writer.add_scalar("Loss/train_withpe" + dataname+str(mf), g_loss, epoch)

        val_losses = {}
        for i in loaders:
            if i.startswith("val"):
                val_loss=val_epoch(
                    G = G,
                    device=device, 
                    dataloader=loaders[i],
                    )
                val_losses[i] = val_loss
                print(f"{mf}_{i} loss: {val_loss}",end='')
        writer.add_scalars("Loss/val_withpe" + dataname+str(mf), val_losses, epoch)
        writer.flush()
        val_loss_list = [val_losses[i] for i in val_losses]
        metric = np.mean(val_loss_list)
        if metric < global_min:
            global_min = metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': G.state_dict(),
                'optim_state_dict': opt_g.state_dict(),
                'loss': g_loss
            }, dataname + "_deepdr")

def main(conf_file="conf_deepdr.yml"):
    ### Load config
    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)

    num_epochs = conf["epochs"]
    lr=conf["lr"][dataname]
    batch_size = conf["batch_size"][dataname]
    activation=conf["activation"]
    feature_in=conf["feature_in"][dataname]
    feature_hid=conf["feature_hid"][dataname]
    heads=conf["heads"][dataname]
    num_layers=conf["num_layers"][dataname]
    mf=conf["mask_frac"][dataname]

    config = {
        "bs": batch_size,
        "layers": num_layers,
        "heads": heads,
        "hid": feature_hid,
        "mask_frac": mf,
        "lr": lr,
        "feature_in":feature_in, 
        "ac":activation,
        "num_epochs":num_epochs,
    }
    unique_msg = f'{config}'
    print(unique_msg)
    writer = SummaryWriter('runs/' + dataname + '_deepdr/'+ unique_msg)
    train(config, writer, checkpoint_dir=modelpath)

import argparse
import threading
from multiprocessing.dummy import Pool
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str)
parser.add_argument('--modelpath', type=str, default=None)
args = parser.parse_args()
dataname = args.dataname
modelpath = args.modelpath
main()
