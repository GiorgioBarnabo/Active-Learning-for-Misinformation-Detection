import sys, os

import numpy as np
import random
from sklearn import metrics

from torch.utils.data import random_split, ConcatDataset, Subset
from torch_geometric.data import DataLoader, DataListLoader

sys.path.insert(1, os.getcwd()+'/graph_utils')
import data_loader

def load_all_ordered(path, warm_start_years, training_years, model_type, graph_test_size, graph_feature):
    graph_dataset = None
    if model_type=="time":
        all_loaded_data = {}
        all_eval_data = {}
        for filename in os.listdir(path):
            app = filename.split("_")
            if len(app)==1: #x.npy or others
                continue
            elif "eval" in filename:
                is_eval = True
                if "-" in filename:
                    _,file_year_month,is_y = app
                else:
                    _,is_y = app
                    file_year_month = "ALL_EVAL"
            else:
                is_eval = False
                file_year_month,is_y = app
                
            if "-" in file_year_month: #to avoid evaluation_x/... files
                app = file_year_month.split("-")
                file_year = int(app[0])
                file_month = int(app[1])
            else:
                file_year = np.inf
                file_month = np.inf

            if is_eval or (file_year >= warm_start_years[0] and file_year <= warm_start_years[1]) or (file_year >= training_years[0] and file_year <= training_years[1]):
                is_y = 1*(is_y.replace(".npy","") == "y") #check if is y

                #print(filename)
                new_load = np.load(os.path.join(path, filename))
                new_load = new_load.astype('float32') #convert type

                if new_load.shape[0]>1: #some files are empty.....
                    #SHUFFLE DATA
                    pos = np.arange(new_load.shape[0])
                    random.Random(123).shuffle(pos)
                    new_load = new_load[pos]

                    if is_eval:
                        if file_year_month not in all_eval_data:
                            all_eval_data[file_year_month] = [None,None] #First is x, second is y
                    
                        all_eval_data[file_year_month][is_y] = new_load
                    else:
                        if file_year_month not in all_loaded_data:
                            all_loaded_data[file_year_month] = [None,None] #First is x, second is y
                    
                        all_loaded_data[file_year_month][is_y] = new_load
    elif model_type=="graph":
        app = path.split("/")
        graph_data_folder = "/".join(app[:-1])
        dataname = app[-1]

        graph_dataset = data_loader.FNNDataset(root=graph_data_folder, feature=graph_feature, empty=False, name=dataname, transform=data_loader.ToUndirected())

        num_test = int(len(graph_dataset) * graph_test_size)
        num_training = len(graph_dataset) - num_test
        training_set, test_set = random_split(graph_dataset, [num_training, num_test])

        all_loaded_data ={"2020": training_set}
        all_eval_data = {"ALL_EVAL": test_set}

    all_year_month_ordered_keys = sorted(all_loaded_data.keys())
    all_year_month_test_ordered_keys = sorted(all_eval_data.keys())

    '''
    if all_year_month_ordered_keys == all_year_month_test_ordered_keys:
        print("SAME MONTHS")
    else:
        print("!!!!!! ERROR - NOT SAME MONTHS!!!!!!!!") #### CAN HAPPEN IF TRAIN/TEST HAVE EMPTY MONTHS
    '''

    return all_year_month_ordered_keys, all_loaded_data, all_year_month_test_ordered_keys, all_eval_data, graph_dataset

def get_warm_start_key_ids(all_year_month_ordered_keys, warm_start_years):
    for starting_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>=warm_start_years[0]: #until warm_start_year INCLUDED
            break

    for current_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>warm_start_years[1]: #until warm_start_year INCLUDED
            break
    
    return starting_key_id,current_key_id

def prepare_new_data_in_range(all_data, seq_len, data_opt, all_year_month_ordered_keys, starting_key_id, current_key_id, model_type):
    new_x, new_y = get_new_data_in_range(all_data, all_year_month_ordered_keys, starting_key_id, current_key_id, model_type)
    
    if model_type=="time":
        new_x = prepare_data(new_x, seq_len, data_opt)

    return new_x, new_y

def get_new_data_in_range(all_data, all_year_month_ordered_keys, starting_key_id, current_key_id, model_type):
    new_x = []
    new_y = []
    for key in all_year_month_ordered_keys[starting_key_id:current_key_id]:
        if model_type=="time":
            new_x.append(all_data[key][0])
            new_y.append(all_data[key][1])
        elif model_type=="graph":
            new_x.append(all_data[key])

    if model_type=="time":
        new_x = np.concatenate(new_x)
        new_y = np.concatenate(new_y)

        pos = np.arange(new_x.shape[0])
        random.Random(123).shuffle(pos)
        new_x = new_x[pos]
        new_y = new_y[pos]

    elif model_type=="graph":
        new_x = ConcatDataset(new_x)

        pos = np.arange(len(new_x))
        random.Random(123).shuffle(pos)
        new_x = Subset(new_x, pos)

    return new_x, new_y

def prepare_test_data(all_test_data,seq_len,data_opt):
    prepared_test_data = {}
    for key,(x,y) in all_test_data.items():
        if x.shape[0]>1: #some tests are empty.....
            prepared_test_data[key] = [prepare_data(x,seq_len,data_opt),y]

    all_nonempty_year_month_test_ordered_keys = sorted(prepared_test_data.keys())

    return all_nonempty_year_month_test_ordered_keys, prepared_test_data


def model_evaluate_per_month(model, all_nonempty_year_month_test_ordered_keys, test_data, model_type):
    cut_off = 0.5
    rs = []
    for key in all_nonempty_year_month_test_ordered_keys: 
        if model_type=="time":
            x,y_test = test_data[key]
            pred_test = np.array(model.predict(x))>=cut_off #[:,1] should be the same
        else:
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