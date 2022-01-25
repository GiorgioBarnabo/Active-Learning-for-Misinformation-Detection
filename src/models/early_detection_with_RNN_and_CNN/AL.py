import sys
sys.path.append('..')

import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch_geometric.data import DataLoader, DataListLoader
from torch.utils.data import random_split, ConcatDataset

from graph_model import compute_test

def keep_sum_vec(vec, sm):
    ##### THIS FUNCTION IS JUST TO KEEP THE SUM OF vec EQUAL TO sm
    new_tot = np.sum(vec.astype(int))
    missing = sm-new_tot
    if missing != 0:
        #print("MISSING:", missing)
        app = np.ones((len(vec),))*missing//len(vec)
        rest = missing%len(vec)
        app[:rest] += 1
        #if np.sum(rest) == missing:
        #    print("YESSSS")
        vec += app
    #######
    vec = vec.astype(int)
    return vec

def merge_new_data(data, new_x, new_y, train_val_test_split_ratio,
                   AL_method, keep_all_new,
                   val_offline_num_urls, offline_AL,
                   model, model_type,
                   num_urls_k, combined_AL_nums, diversity_nums,
                   train_last_samples, val_last_samples, add_val_to_train):
    if keep_all_new: #data['x_train'] is None:
        new_ids = None
    else:
        if len(new_x)<=num_urls_k: #x
            new_ids = None
        else:
            if val_offline_num_urls>0: #always random, maybe change?
                print("HOPEFULLY GET VAL ONCE")
                val_ids = AL_random(val_offline_num_urls)

                if model_type=="time":
                    val_x,val_y = new_x[val_ids],new_y[val_ids]
                elif model_type=="graph":
                    val_x = torch.utils.data.Subset(new_x, val_ids)
                    val_y = None

                if data['x_valid'] is None: #First batch
                    data['x_valid'] = val_x
                    data['y_valid'] = val_y
                else:
                    print("!!!!!!THIS SHOULDN'T HAPPEN!!!!!!")
                    if model_type=="time":
                        data['x_valid'] = np.concatenate([data['x_valid'],val_x])
                        data['y_valid'] = np.concatenate([data['y_valid'],val_y])
                    elif model_type=="graph":
                        data['x_valid'] = ConcatDataset([data['x_valid'],val_x])
                        data['y_valid'] = None

                if model_type=="time":
                    new_x = np.delete(new_x, val_ids, axis=0)
                    new_y = np.delete(new_y, val_ids, axis=0)
                elif model_type=="graph":
                    new_range = np.array(list(set(range(len(new_x))).difference(set(list(val_ids)))))
                    new_x = torch.utils.data.Subset(new_x, new_range)
        
            take_until = min(len(new_x),num_urls_k)

            if AL_method == "random":
                new_ids = AL_random(take_until)
            elif AL_method == "uncertainty-margin":
                new_ids = AL_uncertainty_margin(new_x, take_until, model, model_type)
            elif AL_method == "diversity-cluster":
                new_ids = AL_diversity_cluster(new_x, take_until, diversity_nums)
            elif AL_method == "combined":
                new_ids_divided = []
                AL_nums_divided = []
                if combined_AL_nums[0]>0:
                    new_ids_divided.append(AL_random(take_until))
                    AL_nums_divided.append(combined_AL_nums[0])
                if combined_AL_nums[1]>0:
                    new_ids_divided.append(AL_uncertainty_margin(new_x, take_until, model, model_type))
                    AL_nums_divided.append(combined_AL_nums[1])
                if combined_AL_nums[2]>0:
                    new_ids_divided.append(AL_diversity_cluster(new_x, take_until, diversity_nums))
                    AL_nums_divided.append(combined_AL_nums[2])
                
                #print(new_ids_divided)

                new_ids = {}
                i = 0
                i123 = [0 for _ in range(len(new_ids_divided))]
                while len(new_ids)<take_until:
                    poss = new_ids_divided[i][i123[i]:(i123[i]+AL_nums_divided[i])]
                    new_ids.update(dict(zip(poss,[None]*len(poss))))

                    i123[i] += AL_nums_divided[i]
                    i = (i+1) % len(new_ids_divided)
                    #print(new_ids.keys())

                new_ids = np.array(list(new_ids.keys()))
                #print(new_ids)
                if len(new_ids)>take_until:
                    new_ids = new_ids[:take_until]

                if len(new_ids) != take_until:
                    #print(np.sum(samples_per_AL))
                    #print(take_until)
                    print("!!!"*10," COMBINED NOT WORKING? ","!!!"*10)
            else:
                print("NOT IMPLEMENTED YET")

    rem_x = None
    rem_y = None
    if new_ids is not None:
        if offline_AL>0:
            if model_type=="time":
                rem_x = np.delete(new_x, new_ids, axis=0)
                rem_y = np.delete(new_y, new_ids, axis=0)
            elif model_type=="graph":
                rem_range = np.array(list(set(range(len(new_x))).difference(set(list(new_ids)))))
                rem_x = torch.utils.data.Subset(new_x, rem_range)
        
        if model_type=="time":
            new_x,new_y = new_x[new_ids],new_y[new_ids]
        elif model_type=="graph":
            new_x = torch.utils.data.Subset(new_x, new_ids)

    if model_type=="time":
        new_positives = np.sum(new_y)
        new_negatives = len(new_y) - new_positives
    elif model_type=="graph":
        if model.multi_gpu:
            loader = DataListLoader
        else:
            loader = DataLoader
        app = loader(new_x, batch_size=model.batch_size)
        cont = 0
        for dat in app:
            cont+=np.sum(dat.y.cpu().detach().numpy())
        new_positives = cont
        new_negatives = len(new_x) - new_positives

    if model_type=="time":
        if offline_AL>0:
            new_x_train,new_y_train = new_x, new_y
            new_x_valid,new_y_valid = None,None
        else:
            new_x_train,new_y_train,new_x_valid,new_y_valid = split_by_ratio(new_x, new_y, train_val_test_split_ratio) 
    elif model_type=="graph":
        if offline_AL>0:
            new_x_train = new_x
            new_x_valid, new_y_train, new_y_valid = None,None,None
        else:
            num_train = int(train_val_test_split_ratio[0]*len(new_x))
            num_val = len(new_x)-num_train
            new_x_train, new_x_valid = random_split(new_x, [num_train, num_val])
            new_y_train,new_y_valid = None,None

    #print(new_x_train.shape,new_x_valid.shape)
    if data['x_train'] is None: #First batch
        data['x_train'] = new_x_train
        data['y_train'] = new_y_train
        
        if new_x_valid is not None:
            data['x_valid'] = new_x_valid
            data['y_valid'] = new_y_valid
    else:
        if new_x_valid is not None:
            if model_type=="time":
                data['x_valid'] = np.concatenate([data['x_valid'],new_x_valid])
                data['y_valid'] = np.concatenate([data['y_valid'],new_y_valid])
            elif model_type=="graph":
                data['x_valid'] = ConcatDataset([data['x_valid'],new_x_valid])
                data['y_valid'] = None

            if val_last_samples!=np.inf:
                app_x = data['x_valid'][:-val_last_samples]
                app_y = data['y_valid'][:-val_last_samples]

                if add_val_to_train:
                    data['x_train'] = np.concatenate([data['x_train'],app_x])
                    data['y_train'] = np.concatenate([data['y_train'],app_y])
                
                data['x_valid'] = data['x_valid'][-val_last_samples:]
                data['y_valid'] = data['y_valid'][-val_last_samples:]
            
        if model_type=="time":
            data['x_train'] = np.concatenate([data['x_train'],new_x_train])
            data['y_train'] = np.concatenate([data['y_train'],new_y_train])
        elif model_type=="graph":
            data['x_train'] = ConcatDataset([data['x_train'],new_x_train])
            data['y_train'] = None      
        
        if train_last_samples!=np.inf:
            data['x_train'] = data['x_train'][-train_last_samples:]
            data['y_train'] = data['y_train'][-train_last_samples:]
            #print(data['x_train'].shape,data['y_train'].shape)
            
        #data['x_test'] = np.concatenate([data['x_test'],new_x_test])
        #data['y_test'] = np.concatenate([data['y_test'],new_y_test])

    return data, (rem_x,rem_y), (new_positives, new_negatives)

def split_by_ratio(new_x, new_y, train_val_test_split_ratio):
    app = int(new_x.shape[0] * train_val_test_split_ratio[0])

    new_x_train = new_x[:app]
    new_y_train = new_y[:app]
    
    new_x_valid = new_x[app:]
    new_y_valid = new_y[app:]

    #new_x_valid = new_x[int(new_x.shape[0] * train_val_test_split_ratio[0]): int(new_x.shape[0] * train_val_test_split_ratio[0]) + int(new_x.shape[0] * train_val_test_split_ratio[1]), :]
    #new_y_valid = new_y[int(new_x.shape[0] * train_val_test_split_ratio[0]): int(new_x.shape[0] * train_val_test_split_ratio[0]) + int(new_x.shape[0] * train_val_test_split_ratio[1])]

    #new_x_test = new_x[int(n * train_val_test_split_ratio[0]) + int(n * train_val_test_split_ratio[1]):, :]
    #new_y_test = new_y[int(n * train_val_test_split_ratio[0]) + int(n * train_val_test_split_ratio[1]):]

    return new_x_train,new_y_train,new_x_valid,new_y_valid #,new_x_test,new_y_test

def AL_random(take_until):
    app = np.array(range(take_until))
    return app

def AL_uncertainty_margin(new_x, take_until, model, model_type):
    if model_type=="time":
        model_input = new_x
        pred_y = model.predict(model_input)[:,0] #[:,1] should be the same
    else:
        if model.multi_gpu:
            loader = DataListLoader
        else:
            loader = DataLoader

        print("MODEL INPUT")

        #model_input = loader(new_x, batch_size=model.batch_size)
        model_input = loader(new_x, batch_size=len(new_x))

        print("MODEL PREDICT")

        pred_y = model.predict(model_input)

        print("FINISHED")

    from_most_uncertain_ids = np.argsort(np.abs(pred_y-0.5))
    most_uncertain_ids = from_most_uncertain_ids[:take_until]

    return most_uncertain_ids

def AL_diversity_cluster(new_x, take_until, diversity_nums):
    flatten_x = new_x.reshape((new_x.shape[0],new_x.shape[1] * new_x.shape[2]))

    flatten_x /= np.sqrt((flatten_x**2).sum(axis=1))[:,None] #normalize X to use cosine_similarity

    #print("SAMPLES",flatten_x.shape[0])
    #print("TU",take_until)
    #print("DIV_NUMS",diversity_nums)
    num_clusters = max(take_until//np.sum(diversity_nums),1) #at least 1
    #print("num_clusters",num_clusters)

    cluster_result = KMeans(n_clusters = num_clusters, random_state=0).fit(flatten_x)

    cluster_centers = cluster_result.cluster_centers_

    centroids_ids = []
    outliers_ids = []
    random_ids = []

    remaining_per_cluster = []
    ids_remaining_per_cluster = []
    for i in range(num_clusters):
        idx_cluster_i = np.where(cluster_result.labels_==i)[0]
        #print("IDS:",idx_cluster_i)

        remaining_per_cluster.append(max(len(idx_cluster_i)-np.sum(diversity_nums),0))
        #print("REMS",remaining_per_cluster)

        if remaining_per_cluster[-1]==0:
            random_ids += list(idx_cluster_i)
            ids_remaining_per_cluster.append(None)
        else:
            cos_sims = cosine_similarity(flatten_x[idx_cluster_i],cluster_centers[i:(i+1)])[:,0]
            sorted_idx = np.argsort(cos_sims)
    
            centroids_ids += list(idx_cluster_i[sorted_idx[:diversity_nums[0]]])
            outliers_ids += list(idx_cluster_i[sorted_idx[-diversity_nums[1]:]])

            random_poss = idx_cluster_i[sorted_idx[diversity_nums[0]:-diversity_nums[1]]]

            random.Random(123).shuffle(random_poss)

            random_ids += list(random_poss[:diversity_nums[2]])
            ids_remaining_per_cluster.append(random_poss[diversity_nums[2]:])

    #print("CENTR:",centroids_ids)
    #print("OUTS:",outliers_ids)
    #print("RANDS:",random_ids)

    remaining = take_until - len(centroids_ids) - len(outliers_ids) - len(random_ids)
    #print("REM",remaining)

    #cont = 0
    while remaining>0:# and cont<10:
        #print("REM X CLUST",remaining_per_cluster)
        clusters_with_remaining = np.where(np.array(remaining_per_cluster)>0)[0]
        #print("clusters_with_remaining",clusters_with_remaining)

        from_each_cluster = np.ones((len(clusters_with_remaining))) * remaining / len(clusters_with_remaining)
        
        #print("FEC",from_each_cluster)
        from_each_cluster = keep_sum_vec(from_each_cluster, remaining)
        #print("FEC",from_each_cluster)

        app = np.where(from_each_cluster>0)[0]
        from_each_cluster = from_each_cluster[app]
        clusters_with_remaining = clusters_with_remaining[app]
        #print("from_each_cluster",from_each_cluster)
        #print("clusters_with_remaining",clusters_with_remaining)

        for i,num in zip(clusters_with_remaining,from_each_cluster):
            #print("i,num",i,num)
            random_poss = ids_remaining_per_cluster[i]
            #print("RP-len",len(random_poss))

            if len(random_poss)>=num:
                random_ids += list(ids_remaining_per_cluster[i][:num])

                remaining -= num
                remaining_per_cluster[i] -= num
                ids_remaining_per_cluster[i] = ids_remaining_per_cluster[i][num:]
            else:
                random_ids += list(ids_remaining_per_cluster[i])

                remaining -= len(ids_remaining_per_cluster[i])
                remaining_per_cluster[i] = 0
                ids_remaining_per_cluster[i] = None
        #cont+=1
        #print("END REM",remaining)

    #if cont==10:
        #print(OKANDLAKNDALKN)
    #centroids = np.argmin(np.linalg.norm(np.expand_dims(flatten_x,-1) - np.expand_dims(np.transpose(),0), axis=1), axis=0)
    #centroids = np.unique(centroids)

    #print("CENTR:",centroids_ids)
    #print("OUTS:",outliers_ids)
    #print("RANDS:",random_ids)

    to_take_ids = np.array(centroids_ids + outliers_ids +random_ids)

    if len(to_take_ids) < take_until:
        print("!!!"*10," CLUSTERING CENTROIDS NOT OK? ","!!!"*10)

    return to_take_ids