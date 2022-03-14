import argparse
import time
from tqdm import tqdm
from copy import deepcopy

import torch
import pickle
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import sys
# insert at 1, 0 is the script path (or '' in REPL)
import os

os.chdir('/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src')
sys.path.insert(1, '')
#sys.path.append("/Online-Active-Learning-for-Misinformation-Detection/src/models/GNN-FakeNews/")

print(os.getcwd())
print(sys.path)

from utils.data_loader import *
from utils.eval_helper import *

project_folder = os.path.join('../../../')


dataset_name = "condor"

dataset = FNNDataset(root='../data', feature='bert', empty=False, name=dataset_name, transform=ToUndirected())

loader = DataLoader

split_ratio = [0.60, 0.10, 0.20]

num_training = int(len(dataset) * split_ratio[0])
num_val = int(len(dataset) * split_ratio[1])
num_test = len(dataset) - (num_training + num_val)

train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=num_test)
train_idx, val_idx = train_test_split(train_idx, test_size=num_val)

training_set = Subset(dataset, train_idx)
validation_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)

with open("../data/{}/train_val_test_graphs/training_graph.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(training_set, f)

with open("../data/{}/train_val_test_graphs/train_idx.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(train_idx, f)

with open("../data/{}/train_val_test_graphs/validation_graph.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(validation_set, f)

with open("../data/{}/train_val_test_graphs/val_idx.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(val_idx, f)

with open("../data/{}/train_val_test_graphs/test_graph.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(test_set, f)

with open("../data/{}/train_val_test_graphs/test_idx.pickle".format(dataset_name), 'wb') as f:
    pickle.dump(test_idx, f)

'''
num_training = int(len(dataset) * split_ratio[0])

num_val = int(len(dataset) * split_ratio[1])
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

print(type(training_set))
print(type(training_set[0]))
print(training_set[0].num_nodes)
print(training_set[0].num_edges)

train_loader = loader(training_set, batch_size=1, shuffle=True)
val_loader = loader(validation_set, batch_size=30, shuffle=False)
test_loader = loader(test_set, batch_size=30, shuffle=False)

for graph in train_loader:
    print(graph.edge_index)
    print(graph.x.shape)
    print(graph.y)
    print(type(graph))
    break
'''