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

# # # pytorch lightning
from torch_geometric.datasets import Planetoid,WebKB,WikipediaNetwork
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
import torch_cluster
from torch_geometric.data import Data
from torch_geometric.utils import degree

def calc_random_walk_matrix(edge_index,walk_len = 9):
    r = len(edge_index[0])
    N = edge_index.max() + 1
    adj = torch.sparse_coo_tensor(edge_index, [1.] * r, (N, N))

    t = degree(edge_index[0],num_nodes=N)
    t = 1. / t
    n = len(t)
    ind = torch.tensor([[i, i] for i in range(n)]).T
    diag = torch.sparse_coo_tensor(ind, t, (n, n)).float()
    random_walk = torch.sparse.mm(diag,adj)
    rw=[random_walk]
    for _ in range(walk_len):
        out = torch.sparse.mm(rw[-1], random_walk)
        rw.append(out)
    return rw



# NOTE: p and q are the parameters of node2vec, need to be tuned [(1,1),(1,0.1),(0.1,1),(0.1,0.1),(0.5,0.5)]   [(0.5,0.05),(0.05,0.5)]
def pack_multiple_walk(dataset,walk_length,num_walks):
    res = []
    edge_index = dataset.edge_index
    N = dataset.num_nodes
    X = dataset.x

    # setting !!!!!!!!!!!####################
    # p=set_p
    # q=set_q
    p=0.5
    q=0.05


    model = Node2Vec(edge_index, embedding_dim=5, walk_length=walk_length,
         context_size=1, walks_per_node=num_walks,
         num_negative_samples=1, p=p, q=q, sparse=True)
    loader = model.loader(batch_size=1, shuffle=False)
    for pos,_ in loader:
        pos = pos.view(walk_length,num_walks).T.unsqueeze(0)
        res.append(pos)
    node_images = torch.cat(res,dim=0)
    node_images_feature = X[node_images]
    node_images_feature = node_images_feature
    return node_images,node_images_feature


def gen_node_image(list_random_walks,num_nodes):
    # a list of random walks: [(node1_walk1,node2_walk1,...),(node1,walk2,node2,walk2,....)]
    node_image = []
    for i in range(num_nodes):
        node_seq = torch.tensor([w[i].numpy().tolist() for w in list_random_walks]).long()
        node_image.append(node_seq) # node image is: [node1_img,node2_img,.....]
    return node_image

def gen_feature_from_img(node_imgs,X,num_nodes,k_hop_dict):
    node_struct_features=[]
    # node_imgs is a multi-dimensional tensor
    node_feature = X[node_imgs]
    for i in range(num_nodes):
        f = add_struct_feature(node_imgs[i],i,num_nodes,k_hop_dict)
        node_struct_features.append(f)
    node_struct_features = torch.cat([i.unsqueeze(0) for i in node_struct_features], dim=0)
    return node_feature,node_struct_features


def add_struct_feature(node_image,index,num_nodes,k_hop_dict):  # 4-hop as default config
    feats = torch.tensor([[0,0,0,0,0,1]]*num_nodes).float()
    feats[index] = torch.tensor([1,0,0,0,0,0]).float()
    for idx,k_hop in enumerate(k_hop_dict):
        if index not in k_hop:
            continue
        vals = k_hop[index]
        if idx==0:
            f = torch.tensor([0,1,0,0,0,0]).float()
        elif idx==1:
            f = torch.tensor([0, 0, 1, 0, 0, 0]).float()
        elif idx==2:
            f = torch.tensor([0, 0, 0, 1, 0, 0]).float()
        else:
            f = torch.tensor([0, 0, 0, 0, 1, 0]).float()
        for v in vals:
            feats[v]=f
    node_image_struct_feats = feats[node_image]
    return node_image_struct_feats


def load_data_from_disk(fpath,index,num_walks):
    file_name = fpath+'/data_point_'+str(index)+'.pt'
    res = torch.load(file_name)
    img = res['img'][:,:num_walks,:]
    feature = res['feature'][:,:,:num_walks,:]
    struct_feature = res['struct_feature'][:, :, :num_walks, :]
    return (img,feature,struct_feature)

def load_data(dset_name,walk_length,walk_num):
    node_struct_features = []
    # dset = Planetoid(root='dataset/', name=dset_name)

    # dset = WebKB(root='dataset/', name=dset_name)

    dset = WikipediaNetwork(root='dataset/', name=dset_name)

    # print(dset[0])
    processor = EdgeIndex_Processor(edge_index=dset[0].edge_index)
    k_hop_neibrs = processor.run([2, 3, 4])
    node_imgs, node_imgs_feature = pack_multiple_walk(dset[0], walk_length, walk_num)
    N = len(node_imgs)
    num_nodes = dset[0].num_nodes

    for i in range(N):
        node_struct_feats = add_struct_feature(node_imgs[i], i, num_nodes, k_hop_neibrs)
        node_struct_features.append(node_struct_feats)
    # print (len(node_imgs),len(node_imgs_feature),len(node_struct_features))
    # print (node_imgs[0].shape,node_imgs_feature[0].shape,node_struct_features[0].shape)
    node_imgs = torch.cat([i.unsqueeze(0) for i in node_imgs], dim=0)
    node_imgs_feature = torch.cat([i.unsqueeze(0) for i in node_imgs_feature], dim=0)
    node_struct_features = torch.cat([i.unsqueeze(0) for i in node_struct_features], dim=0)
    val = k_hop_neibrs[-1]

    isolated_node_tag = [0]*N  # 标记一个节点是否是在小的community里面，1：不是，0：是
    for v in val:
        isolated_node_tag[v]=1
    isolated_node_tag = torch.tensor(isolated_node_tag).long()

    # print(node_imgs.shape)
    # print(node_imgs_feature.shape)
    # print(node_struct_features.shape)
    node_imgs_feature = torch.permute(node_imgs_feature,(0,3,1,2))
    node_struct_features = torch.permute(node_struct_features,(0,3,1,2))
    return node_imgs,node_imgs_feature,node_struct_features,isolated_node_tag,dset[0],k_hop_neibrs

def resample_data(dset,k_hop_neibrs,walk_len,walk_num):
    node_struct_features = []
    num_nodes = dset.num_nodes
    node_imgs, node_imgs_feature = pack_multiple_walk(dset, walk_len, walk_num)
    for i in range(num_nodes):
        node_struct_feats = add_struct_feature(node_imgs[i], i, num_nodes, k_hop_neibrs)
        node_struct_features.append(node_struct_feats)
    node_imgs = torch.cat([i.unsqueeze(0) for i in node_imgs], dim=0)
    node_imgs_feature = torch.cat([i.unsqueeze(0) for i in node_imgs_feature], dim=0)
    node_struct_features = torch.cat([i.unsqueeze(0) for i in node_struct_features], dim=0)
    val = k_hop_neibrs[-1]

    # isolated_node_tag = [0]*num_nodes  # 标记一个节点是否是在小的community里面，1：不是，0：是
    # for v in val:
    #     isolated_node_tag[v]=1
    # isolated_node_tag = torch.tensor(isolated_node_tag).long()
    node_imgs_feature = torch.permute(node_imgs_feature,(0,3,1,2))
    node_struct_features = torch.permute(node_struct_features,(0,3,1,2))
    return node_imgs,node_imgs_feature,node_struct_features


class EdgeIndex_Processor():
    def __init__(self, edge_index):
        super().__init__()
        adj,N = self.to_sparse_tensor(edge_index)
        adj_with_selfloop = self.to_sparse_tensor_with_selfloop(edge_index)
        self.N = N
        self.adj = adj.float()
        self.adj_with_loop = adj_with_selfloop.float()
        self.k_hop_neibrs = [adj.float()]

    def to_sparse_tensor(self, edge_index):
        edge_index = remove_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t, N

    def to_sparse_tensor_with_selfloop(self, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t


    def calc_adj_power(self,adj, power):
        t = adj
        for _ in range(power - 1):
            t = torch.sparse.mm(t, adj)
        # set value to one
        indices = t.coalesce().indices()
        v = t.coalesce().values()
        v = torch.tensor([1 if i > 1 else i for i in v])
        diag_mask = indices[0] != indices[1]
        indices = indices[:, diag_mask]
        v = v[diag_mask]
        t = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return t

    def postprocess_k_hop_neibrs(self,sparse_adj):
        diag = torch.diag(1. / sparse_adj.to_dense().sum(dim=1))
        diag = diag.to_sparse()
        out = torch.sparse.mm(diag, sparse_adj)
        return out


    def calc_k_hop_neibrs(self,k_hop=2):
        adj_hop_k = self.calc_adj_power(self.adj, k_hop)
        one_hop = self.k_hop_neibrs[0]
        prev_hop = self.k_hop_neibrs[1:k_hop]
        for p in prev_hop:
            one_hop += p
        final_res = adj_hop_k - one_hop

        indices = final_res.coalesce().indices()
        v = final_res.coalesce().values()
        v = [0 if i <= 0 else 1 for i in v]
        masking = []
        v_len = len(v)
        for i in range(v_len):
            if v[i] > 0:
                masking.append(i)
        v = torch.tensor(v)
        masking = torch.tensor(masking).long()
        indices = indices[:, masking]
        v = v[masking]
        final_res = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return final_res

    def adj_to_dict(self,adj):
        res = {}
        df = pd.DataFrame(adj.coalesce().indices().T.numpy()).groupby([0],as_index=False).agg({1:list})
        for _,r in df.iterrows():
            k = r[0]
            v = r[1]
            res[k]=v
        return res

    def run(self,k_hop=[2,3,4]):
        dicts = []
        for k in k_hop:
            t = self.calc_k_hop_neibrs(k)
            self.k_hop_neibrs.append(t.float())
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        for o in self.k_hop_neibrs:
            dicts.append(self.adj_to_dict(o))
        return dicts


if __name__=='__main__':
    import time
    from tqdm import tqdm
    # print('start')
    # s = time.time()


    # set_p=0.5
    # set_q=0.05
    # print('set_p:',set_p,'set_q:',set_q)

    save_path = 'data/squirrel_p0.05_q0.5'
    print('save_path:',save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    def task(i):
        print (f'current progress:{i} for generating data...................')
        node_imgs, node_imgs_feature, node_struct_features, isolated_node_tag, dset, k_hop_neibrs = load_data('squirrel',10,10)
        s = {'img':node_imgs,'strcuture:':node_struct_features}
        torch.save(s,save_path+f'/data_point_{i}.pt')

    from multiprocessing import Pool
    import os


    data_inputs=range(500)

    pool = Pool(10)                         # Create a multiprocessing Pool
    pool.map(task, data_inputs)  # process data_inputs iterable with pool



    # for i in tqdm(range(500)):
    #     # print (f'current progress:{i} for generating data...................')
    #     # node_imgs, node_imgs_feature, node_struct_features, isolated_node_tag, dset, k_hop_neibrs = load_data('Texas',10,10)
    #     node_imgs, node_imgs_feature, node_struct_features, isolated_node_tag, dset, k_hop_neibrs = load_data('squirrel',10,10)
    #     s = {'img':node_imgs,'strcuture:':node_struct_features}
    #     torch.save(s,save_path+f'/data_point_{i}.pt')

