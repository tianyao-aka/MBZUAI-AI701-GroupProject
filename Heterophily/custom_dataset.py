import json
import math
import os
# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchmetrics import AUROC,Accuracy


class Node_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self,root_dir):
        self.dir = root_dir
        self.list = [0]*1
        self.num_walk=10
        self.walk_len = 30
        self.cache=[]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        idx = int(np.random.choice(500))
        if len(self.cache)==0:
            file_name = 'data/Texas_p0.05_q0.5/data_point_' + str(idx) + '.pt'
            res = torch.load(file_name)
            img = res['img'][:, :self.num_walk, :self.walk_len]
            feature = res['feature'][:, :, :self.num_walk, :self.walk_len]
            struct_feature = res['struct_feature'][:, :, :self.num_walk, :self.walk_len]
            self.cache.append((img,feature,struct_feature))
            return img,feature,struct_feature
        else:
            return self.cache[0]


if __name__=='__main__':
    import time
    # s = time.time()
    # for i in range(3):
    #     print (f'current progress:{i} for generating data...................')
    #     node_imgs, node_imgs_feature, node_struct_features, isolated_node_tag, dset, k_hop_neibrs = load_data('Cora',40,20)
    #     s = {'img':node_imgs,'feature':node_imgs_feature,'struct_feature':node_struct_features}
    #     torch.save(s,f'data_point_{i}.pt')
    save_path = 'data/Texas_p0.5'
    x = torch.load(save_path+'/data_point_0.pt')
    print (x['img'][0][:3])
    x = torch.load(save_path+'/data_point_1.pt')
    print ()
    print (x['img'][0][:3])

