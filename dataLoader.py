from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import random
import pickle

# # Dataset definition
class o_m(Dataset):
    def __init__(self, origin, masked):
        self.o = origin
        self.m = masked
    
    def __len__(self):
        return self.o.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.o[idx]), torch.from_numpy(self.m[idx])

# # Make dataset (64:16:20)
def make_dataset_all(dt):

    # reproducibility
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)

    data_o = dt["standard"]

    total_len = len(data_o)
    train_val_len = int(0.8*total_len)
    train_len = int(0.8*train_val_len)
    val_len = train_val_len - train_len

    dataset_all = {}

    for frac in ["30", "40", "50", "60", "70", "80"]:
        # 4:1 for train and test
        data_train_val = o_m(data_o[:train_val_len], dt["scale_mask" + frac][:train_val_len])
        data_test      = o_m(data_o[train_val_len:], dt["scale_mask" + frac][train_val_len:])
        # random split train to train and validation with 4:1
        data_train, data_val = random_split(data_train_val, [train_len,val_len], generator=torch.Generator().manual_seed(5))
        # unmasked data should be consistent
        assert(np.array_equal(data_o[~np.isnan(dt["scale_mask" + frac])], dt["scale_mask" + frac][~np.isnan(dt["scale_mask" + frac])]))

        dataset_all["val"+frac] = data_val
        dataset_all["test"+frac] = data_test

        if "train" not in dataset_all:
            dataset_all["train"] = data_train
        else:
            for i in range(len(data_train)):
                assert(np.array_equal(dataset_all["train"][i][0], data_train[i][0]))
        
    return dataset_all

def make_dataloader(dataname, batch_size=16, num_workers=0):
    # print(dataname)
    if dataname == "ab":
        dataset = pickle.load(open("/home/zhaohuaiyi/GNN-MC/data/AB.dat","rb"))
    elif dataname == "ge":
        dataset = pickle.load(open("/home/zhaohuaiyi/GNN-MC/data/GE.dat","rb"))
    else:
        print("Name not supported")
        exit(1)
    
    data_all = make_dataset_all(dataset)
    train_loader = DataLoader(data_all["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    all_dataloaders = {
        "train": train_loader
    }
    for frac in ["30", "40", "50", "60", "70", "80"]:
        for type in ["val", "test"]:
            all_dataloaders[type + frac] = DataLoader(data_all[type + frac],batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return all_dataloaders




