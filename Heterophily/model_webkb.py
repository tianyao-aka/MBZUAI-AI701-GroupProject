import copy
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

# # pytorch lightning
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


import torch_geometric
import torch_geometric.data as geo_data
import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv,GATConv,GINConv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import time

import argparse

from utils import *
from custom_dataset import Node_Image_Dataset

class Model_grid(nn.Module):
    def __init__(self,in_dim,hidden_dim,struct_dim,out_dim,use_batchnorm=True,first_layer=True):
        super().__init__()
        # self.cls_head = nn.Linear(hidden2_dim*2,num_classes)
        self.struct_emb = nn.Linear(struct_dim,10,bias=False)
        self.first_layer = first_layer
        if first_layer:
            self.conv1by1 = nn.Sequential(nn.Dropout(),nn.Conv2d(in_dim,hidden_dim,kernel_size=(1,1)),nn.ReLU())
            if use_batchnorm:
                self.model = nn.Sequential(nn.Conv2d(hidden_dim+10,32,kernel_size=(3,3),stride=3),nn.AvgPool2d(kernel_size=2,stride=2),
                                           nn.ReLU(),nn.BatchNorm2d(32),nn.Conv2d(32,24,kernel_size=(3,3)),nn.AdaptiveAvgPool2d(1),nn.ReLU(),nn.BatchNorm2d(24),nn.Linear(24,24))
            else:
                self.model = nn.Sequential(nn.Conv2d(hidden_dim+10,32,kernel_size=(3,3),stride=3),nn.AvgPool2d(kernel_size=2,stride=2),
                                           nn.ReLU(),nn.Conv2d(32,24,kernel_size=(3,3)),nn.AdaptiveAvgPool2d(1),nn.ReLU(),nn.Linear(24,out_dim))
        else:
            if use_batchnorm:
                self.model = nn.Sequential(nn.Conv2d(in_dim + 10, hidden_dim, kernel_size=(3, 3), stride=3),
                                           nn.AvgPool2d(kernel_size=2, stride=2),
                                           nn.ReLU(), nn.BatchNorm2d(hidden_dim), nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3)),
                                           nn.AdaptiveAvgPool2d(1), nn.ReLU(), nn.BatchNorm2d(hidden_dim), nn.Linear(hidden_dim, out_dim))
            else:
                self.model = nn.Sequential(nn.Conv2d(hidden_dim + 10, 32, kernel_size=(3, 3), stride=3),
                                           nn.AvgPool2d(kernel_size=2, stride=2),
                                           nn.ReLU(), nn.Conv2d(32, 24, kernel_size=(3, 3)), nn.AdaptiveAvgPool2d(1),
                                           nn.ReLU(), nn.Linear(24, 24))


    def forward(self,node_img_feats):
        # permute img_strcut fature to be from (N,C,H,W) to (N,H,W,C)
        # node_img_struct = torch.permute(node_img_struct,(0,2,3,1))
        # struct = self.struct_emb(node_img_struct)
        # struct = torch.permute(struct,(0,3,1,2))
        if self.first_layer:
            node_img_feats = self.conv1by1(node_img_feats)
        image_feature = torch.cat((node_img_feats,struct),dim=1)  # concat along feature dimension
        out = self.model(image_feature).squeeze()
        return out


class PathGCN(nn.Module):
    def __init__(self,in_dim,walk_len,struct_dim=6,first_layer=0):
        super().__init__()
        self.first_layer = first_layer
        self.strcut_conv = nn.Conv2d(struct_dim,in_dim,groups=1,kernel_size=(1,1),bias=False)
        self.conv = nn.Conv2d(in_dim,in_dim,groups=in_dim,kernel_size=(1,walk_len))
        self.bigram = nn.Conv2d(in_dim,in_dim,groups=in_dim,kernel_size=(2,walk_len))
        self.conv_vertical = nn.Conv2d(in_dim,in_dim,groups=in_dim,kernel_size=(walk_len,1))
        self.linear = nn.Linear(in_dim,in_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_vertical = nn.Linear(in_dim,in_dim)
        self.relu_vertical = nn.ReLU()

        self.linear_bigram = nn.Linear(in_dim,in_dim)
        self.relu_bigram = nn.ReLU()

        if self.first_layer:
            self.position_embedding = nn.parameter.Parameter(torch.randn(in_dim,walk_len))

    def forward(self,x,node_image,node_struct):
        if self.first_layer:
        # intput: x->[N,C] node_image->[N,H,W]; output: [N,C]
            node_struct = self.strcut_conv(node_struct)
            N,C,H,W = node_struct.shape
            pos = self.position_embedding.unsqueeze(0).unsqueeze(2).expand(N,C,H,W).to('cuda:0')
            node_struct = node_struct+pos
        else:
            node_struct = 0.
        h = x[node_image].permute(0,3,1,2)
        h1 = self.conv(h+node_struct).squeeze()
        h1 = torch.mean(h1,dim=-1)  # from (N,C,H) to (N,C)
        h1 = self.relu(self.linear(h1))

        h2 = self.conv_vertical(h+node_struct).squeeze()
        h2 = torch.mean(h2,dim=-1)  # from (N,C,H) to (N,C)
        h2 = self.relu_vertical(self.linear_vertical(h2))

        h3 = self.bigram(h+node_struct).squeeze()
        h3 = torch.mean(h3,dim=-1)  # from (N,C,H) to (N,C)
        h3 = self.relu_bigram(self.linear_bigram(h3))

        h = h1 + h2 + x
        return self.dropout(h)

class Model_Sequence(nn.Module):
    def __init__(self,in_dim,hidden_dim,struct_dim,out_dim,walk_len,layer_num=3):
        super().__init__()
        # self.cls_head = nn.Linear(hidden2_dim*2,num_classes)
        # self.struct_emb = nn.Linear(struct_dim,10,bias=False)
        self.layer_num = layer_num
        self.emb = nn.Sequential(nn.Dropout(),nn.Linear(in_dim,hidden_dim,bias=False),nn.ReLU())
        self.pred = nn.Sequential(nn.Dropout(),nn.Linear(hidden_dim,out_dim))
        self.pathgcn_module = nn.ModuleList()
        for q in range(layer_num):
            if q==0:
                self.pathgcn_module.append(PathGCN(hidden_dim,walk_len,struct_dim=struct_dim,first_layer=True))
            else:
                self.pathgcn_module.append(PathGCN(hidden_dim, walk_len, struct_dim=struct_dim, first_layer=False))


    def forward(self,X,node_image,node_struct):
        # node_img_struct = torch.permute(node_img_struct,(0,2,3,1))
        # struct = self.struct_emb(node_img_struct)
        # struct = torch.permute(struct,(0,3,1,2))
        h = self.emb(X)

        for i in range(self.layer_num):
            h = self.pathgcn_module[i](h,node_image,node_struct)
        h = self.pred(h)
        return h



class PL_Model(pl.LightningModule):
    def __init__(self, in_dim,hidden_dim,struct_dim,num_classes,walk_len = 10,num_walk=10,lr=1e-3,weight_decay=1e-6,dset_name = 'Cora',use_gpu=False,layer_num=3,fpath=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dev = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        print ('use device:',self.dev)
        self.num_classes = num_classes
        self.weight_decay =weight_decay
        self.model = Model_Sequence(in_dim=in_dim,struct_dim=struct_dim,hidden_dim=hidden_dim,out_dim=num_classes,walk_len=walk_len,layer_num=layer_num)
        self.log_prob_nn = nn.LogSoftmax(dim=-1)
        self.acc = Accuracy(top_k=1)
        self.fpath = fpath
        # init data

        _, _, _, _, dset, k_hop_neibrs = load_data(dset_name,walk_len,num_walk)

        # dset = Planetoid(root='dataset/', name='Cora')[0] # no need
        # dset = WebKB(root='dataset/', name=dset_name,)[0]

        # self.node_image,self.node_image_feature,self.node_struct_feature  =load_data_from_disk('data/Cora2',0,num_walk)
        s = torch.load(self.fpath+'data_point_1.pt')
        self.node_img,self.node_struct = s['img'],s['strcuture:']
        self.dataset = dset
        self.X = dset.x
        # self.k_hop_neibrs = k_hop_neibrs
        self.walk_len = walk_len
        self.num_walk = num_walk
        self.val_mask = dset.val_mask[:,0].to(self.dev)
        self.test_mask = dset.test_mask[:,0].to(self.dev)
        self.train_mask = torch.logical_not(torch.logical_or(self.val_mask, self.test_mask))  # fully supervised
        # self.train_mask = dset.train_mask[:,0].to(self.dev)
        self.train_mask =self.train_mask.to(self.dev)
        self.y = dset.y.to(self.dev)
        print ('.......................done loading data...................')


    def forward(self):
        # Forward function that is run when visualizing the graph
        pass


    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,min_lr=1e-3,mode='max',patience=30)
        return {'optimizer':optimizer}

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        #         print (batch_idx)
        # feats,struct_feats,isolated_nodes_ind,y = batch
        self.model.to(self.dev)
        self.log_prob_nn.to(self.dev)
        h = self.model(self.X.to(self.dev),self.node_img.to(self.dev),self.node_struct.to(self.dev))
        h=self.log_prob_nn(h)

        # h = h[self.train_mask]
        # y = y[self.train_mask]
        loss_val = F.nll_loss(h[self.train_mask],self.y[self.train_mask])
        self.log("train_loss", loss_val.item(), prog_bar=True, logger=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        # feats,struct_feats,isolated_nodes_ind,y = batch
        self.model.to(self.device)
        self.log_prob_nn.to(self.device)
        self.acc.to(self.device)
        h = self.model(self.X.to(self.dev),self.node_img.to(self.dev),self.node_struct.to(self.dev))
        h=self.log_prob_nn(h)
        # h = h[self.val_mask]
        # y = y[self.val_mask]
        # print ('h shape',h.shape)
        # print ('y shape',self.y.shape)
        # print ('val mask shape',self.val_mask.shape)
        # print('h[self.val_mask] shape', h[self.val_mask].shape)
        # print('self.y[self.val_mask] shape', self.y[self.val_mask].shape)

        loss_val = F.nll_loss(h[self.val_mask],self.y[self.val_mask])
        acc_val = self.acc(h[self.val_mask],self.y[self.val_mask])
        self.log("val_loss", loss_val.item(), prog_bar=True, logger=True)
        self.log("val_acc", acc_val.item(), prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        # feats,struct_feats,isolated_nodes_ind,y = batch
        self.model.to(self.device)
        self.log_prob_nn.to(self.device)
        self.acc.to(self.device)
        hs = []
        for _ in range(20):
            idx = int(np.random.choice(300))
            s= torch.load(self.fpath + f'data_point_{idx}.pt')
            self.node_img, self.node_struct = s['img'], s['strcuture:']
            h = self.model(self.X.to(self.dev),self.node_img.to(self.dev),self.node_struct.to(self.dev))
            hs.append(h.unsqueeze(1))
        h = torch.mean(torch.cat(hs,dim=1),dim=1)
        h=self.log_prob_nn(h)
        # h = h[self.val_mask]
        # y = y[self.val_mask]
        acc_val = self.acc(h[self.test_mask],self.y[self.test_mask])
        self.log("test_acc", acc_val.item(), prog_bar=True, logger=True)


    def on_train_epoch_start(self):
        # _ = torch.cuda.empty_cache()
        idx = int(np.random.choice(300))
        s = torch.load(self.fpath+f'data_point_{idx}.pt')
        self.node_img, self.node_struct = s['img'], s['strcuture:']


    def on_validation_epoch_end(self):
        pass
        # on each validation epoch ends,resample random walk path
        # if self.cache[0].shape[1]>450:
        #     thres = 0.1
        # elif self.cache[0].shape[1]>200:
        #     thres = 0.2
        # else:
        #     thres = 0.5
        # if np.random.uniform()<thres:
        #     node_img,node_feature,node_struct_feature = resample_data(self.dataset,self.k_hop_neibrs,self.walk_len,self.num_walk)
        #     self.cache[0] = torch.cat((node_img,self.cache[0]),dim=1)
        #     self.cache[1] = torch.cat((node_feature,self.cache[1]), dim=2)
        #     self.cache[2] = torch.cat((node_struct_feature,self.cache[2]),dim=2)
        #     N= self.cache[0].shape[1]
        #     if N>300:
        #         print ('resample size exceeds 300')
        #     if N>500:
        #         self.cache[0] = self.cache[0][:,-500:,:]
        #         self.cache[1] = self.cache[1][:,:, -500:, :]
        #         self.cache[2] = self.cache[2][:, :, -500:, :]
        # rand_indices = torch.randperm(self.total_num_walk)
        # rand_indices = rand_indices[:self.num_walk].long()
        # self.node_image = self.cache[0][:,rand_indices,:]
        # self.node_image_feature = self.cache[1][:,:,rand_indices,:]
        # self.node_struct_feature = self.cache[2][:,:,rand_indices,:]


if __name__=='__main__':
    # model = Model_grid(1433,6)
    # model = Model_Sequence(1433,6,30)
    # x = torch.randn(10,1433,30,30)
    # y = torch.randn(10,6,30,30)
    # x = model(x,y)
    # print (x.shape)

    parser = argparse.ArgumentParser(description='GNN on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--layer_num', type=int, default=2,
                        help='which method to use')

    args = parser.parse_args()
    layer_num = args.layer_num
    x = torch.randn(3, 3)
    t_dataset = torch.utils.data.TensorDataset(x, x)
    t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=3, shuffle=False)
    # fpath ='data/Texas_p0.5_q0.05/'
    fpath ='data/Wisconsin_p0.05_q0.5/'
    print('fpath:',fpath)
    test_results = []
    for _ in range(10):
        # Texas feature_num=1703,num_class=5
        # model = PL_Model(1703, 64, 6, num_classes=5, use_gpu=True, walk_len=10, num_walk=10, layer_num=layer_num,dset_name='Texas',fpath=fpath)
        model = PL_Model(1703, 64, 6, num_classes=5, use_gpu=True, walk_len=10, num_walk=10, layer_num=layer_num,dset_name='Wisconsin',fpath=fpath)
        if torch.cuda.is_available():
            trainer = pl.Trainer(default_root_dir=f'saved_models/',gpus=1, max_epochs=1000,callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),EarlyStopping(monitor='val_acc',mode='max',patience=80)],accelerator='gpu')
            trainer.fit(model=model, train_dataloaders=t_dataloader, val_dataloaders=t_dataloader)
        else:
            trainer = pl.Trainer(default_root_dir=f'saved_models/',gpus=0, max_epochs=800,callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),EarlyStopping(monitor='val_acc',mode='max',patience=50)])
            trainer.fit(model=model, train_dataloaders=t_dataloader, val_dataloaders=t_dataloader)
        best_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        res = trainer.test(model=best_model, dataloaders=t_dataloader)[0]
        print (res)
        test_results.append(res)
    # mean_acc = np.asarray([i['test_acc'] for i in test_results]).mean()
    acc = np.asarray([i['test_acc'] for i in test_results])
    print ('mean acc:',acc.mean(),acc.max())
    print('fpath:',fpath)


