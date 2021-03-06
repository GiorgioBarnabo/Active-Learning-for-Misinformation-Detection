import sys, os

import numpy as np
import random
from sklearn import metrics
import pickle as pkl
from . import gnn_base_models

import torch
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader, DataListLoader

#sys.path.insert(1, os.getcwd()+'/graph_utils')
#sys.path.insert(1,os.path.join("../"*5,"src","our_utils"))
from . import utils
sys.modules['utils'] = utils
sys.modules["new_utils"] = utils 
sys.modules["new_utils.graph_utils"] = utils

def load_graph_data(data_folder, model_name):
    divided_data_folder = os.path.join(data_folder,"train_val_test_graphs")

    app = []
    for data_split_name in ["training","validation","test"]:
        filename = data_split_name+"_graph.pickle"
        if model_name=="bigcn":
            filename = "bigcn_" + filename
        filename = "prova_"+filename
            
        with open(os.path.join(divided_data_folder,filename),"rb") as input_file:
            app.append(pkl.load(input_file))
    
    training_set, validation_set, test_set = app

    all_train_data = {"2020": training_set}
    all_val_data = {"2020": validation_set}
    all_test_data = {"ALL_TEST": test_set}

    return all_train_data, all_val_data, all_test_data


def get_warm_start_key_ids(all_year_month_ordered_keys, warm_start_years):
    for starting_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>=warm_start_years[0]: #until warm_start_year INCLUDED
            break

    for current_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>warm_start_years[1]: #until warm_start_year INCLUDED
            break
    
    return starting_key_id,current_key_id

def get_new_data_in_range(all_data, starting_key_id, current_key_id):
    new_x = []
    for key in list(all_data.keys())[starting_key_id:current_key_id]:
        new_x.append(all_data[key])

    new_x = ConcatDataset(new_x)

    pos = np.arange(len(new_x))
    random.Random(123).shuffle(pos)
    new_x = Subset(new_x, pos)

    return new_x

def model_evaluate_per_month(model, test_data):
    cut_off = 0.5
    rs = []
    for key in test_data.keys():
        if model.multi_gpu:
            loader = DataListLoader
        else:
            loader = DataLoader

        x = test_data[key]
        # model_input = loader(x, batch_size=model.batch_size)
        model_input = loader(x, batch_size=len(x))

        pred_test = model.predict(model_input)>=cut_off

        y_test = []
        for dat in model_input:
            y_test += list(dat.y.cpu().detach().numpy())
        y_test = np.array(y_test)
        
        rs.append(np.array(compute_metrics(pred_test, y_test)))
    return np.array(rs)

def compute_metrics(pred_test, y_test):
    (pre_0,pre_1),(rec_0,rec_1),(f_0,f_1),(_,_) = metrics.precision_recall_fscore_support(y_test,pred_test, labels=[0,1],zero_division=0)
    acc = metrics.accuracy_score(y_test,pred_test)
    #res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)

    try:
        auc = metrics.roc_auc_score(y_test,pred_test)
    except ValueError:
        auc = 0

    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0, auc]

    return res


def compute_new_positives_negatives(model, train_loader):
    #save new_positives/negatives and current_positives/negatives
    cont = 0
    for dat in train_loader:
        cont+=np.sum(dat.y.cpu().detach().numpy())
    current_positives = cont
    current_negatives = len(train_loader.dataset) - current_positives
    return current_positives, current_negatives
