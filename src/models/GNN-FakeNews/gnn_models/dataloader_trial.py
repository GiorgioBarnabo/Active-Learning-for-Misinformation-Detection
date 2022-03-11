import argparse
import time
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader

import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os

print("###############")
print(os.getcwd())
os.chdir('/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src/models/GNN-FakeNews')
print(os.getcwd())
sys.path.insert(1, '')
#sys.path.append("/Online-Active-Learning-for-Misinformation-Detection/src/models/GNN-FakeNews/")

from utils.data_loader import *
from utils.eval_helper import *

project_folder = os.path.join('../../../')


dataset = FNNDataset(root='data', feature='bert', empty=False, name='condor', transform=ToUndirected())

print(type(dataset))

loader = DataLoader

split_ratio = [0.50, 0.20, 0.30]

num_training = int(len(dataset) * split_ratio[0])
num_val = int(len(dataset) * split_ratio[1])
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = loader(training_set, batch_size=1, shuffle=True)
val_loader = loader(validation_set, batch_size=30, shuffle=False)
test_loader = loader(test_set, batch_size=30, shuffle=False)

for graph in train_loader:
    print(graph.edge_index)
    print(graph.x.shape)
    print(graph.y)
    print(type(graph))
    break