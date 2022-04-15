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

print(os.getcwd())

#os.chdir('/home/barnabog/Online-Active-Learning-for-Misinformation-Detection/src')
sys.path.insert(1, '')
#sys.path.append("/Online-Active-Learning-for-Misinformation-Detection/src/models/GNN-FakeNews/")

print(sys.path)

sys.path.insert(1, '../')
from our_utils import utils
sys.modules['utils'] = utils
from our_utils.utils.data_loader import *
from our_utils.utils.eval_helper import *

project_folder = os.path.join('../../')


for dataset_name in ["gossipcop","politifact","condor"]:
    
    if dataset_name!="condor":
        app = dataset_name[:3]
    else:
        app = dataset_name
    
    with open(project_folder+"data/accessory_files/"+app+"_ordered_list_used_in_experiments.pkl", 'rb') as f:
        exp_list = pickle.load(f)
    
        
    for model_name in ["gcn","bigcn"]:
        is_bigcn = model_name=="bigcn"
        
        dataset = FNNDataset(root=project_folder+'/data/graph/', feature='bert', empty=False, name=dataset_name, transform=[ToUndirected(),DropEdge(0.2, 0.2)][is_bigcn])

        dataset.slices["graph_id"] = torch.Tensor(range(len(dataset))).int()
        dataset.data["graph_id"] = exp_list

        '''
        app = ["","bigcn_"][is_bigcn]
        for partition_full_name, partition_short_name in zip(["training","validation","test"],["train","val","test"]):
            with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/"+app+partition_full_name+"_graph.pickle", 'rb') as f:
                partition = pickle.load(f)

            with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/"+app+partition_short_name+"_idx.pickle", 'rb') as f:
                partition_ids = pickle.load(f)
        '''
                        
        split_ratio = [0.60, 0.10, 0.20]

        num_training = int(len(dataset) * split_ratio[0])
        num_val = int(len(dataset) * split_ratio[1])
        num_test = len(dataset) - (num_training + num_val)

        train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=num_test)
        train_idx, val_idx = train_test_split(train_idx, test_size=num_val)

        training_set = Subset(dataset, train_idx)
        validation_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        app = ["","bigcn_"][is_bigcn]

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"training_graph.pickle", 'wb') as f:
            pickle.dump(training_set, f)

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"train_idx.pickle", 'wb') as f:
            pickle.dump(train_idx, f)

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"validation_graph.pickle", 'wb') as f:
            pickle.dump(validation_set, f)

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"val_idx.pickle", 'wb') as f:
            pickle.dump(val_idx, f)

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"test_graph.pickle", 'wb') as f:
            pickle.dump(test_set, f)

        with open(project_folder+"/data/graph/"+dataset_name+"/train_val_test_graphs/prova_"+app+"test_idx.pickle", 'wb') as f:
            pickle.dump(test_idx, f)