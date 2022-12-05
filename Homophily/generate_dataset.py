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
from tqdm import tqdm
import schedule
import psutil
from utils import *
import argparse
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
from torch_geometric.utils import degree



if __name__=='__main__':
    import time
    # s = time.time()
    index = 0
    parser = argparse.ArgumentParser(description='generating dataset')
    parser.add_argument('--dset_name', type=str, default='Cora',
                        help='name of the dataset')
    args = parser.parse_args()
    name = args.dset_name
    def process_node_image():
        global index
        ram = psutil.virtual_memory()[2]
        if ram>65:
            print ('current ram used:',ram)
            return
        if index>100:
            schedule.cancel_job()
            exit()
        else:
            os.system(f'python utils.py --index {index} --dset_name {name} & ')
            print (f'running utils to generate node image for index:{index},current ram usage is:{ram}')
            index +=1

    job = schedule.every(30).seconds.do(process_node_image)

    while True:
        schedule.run_pending()


