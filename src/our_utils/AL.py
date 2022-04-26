import sys
sys.path.append('..')

import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset, TensorDataset #,DataLoader
import pytorch_lightning as pl

from . import graph_model

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

def merge_new_data(current_loaders, 
        current_data, new_data, 
        AL_parameters, keep_all_new, 
        model, trainer, workers_available, 
        batch_size, AL_iteration,
        iteration_of_random_warm_start):
    if keep_all_new: #data['x_train'] is None:
        new_ids = None
    else:
        if len(new_data)<=AL_parameters.num_urls_k: #x
            new_ids = None
        else:
            take_until = min(len(new_data),AL_parameters.num_urls_k)

            if AL_parameters.AL_method == "random" or AL_iteration < iteration_of_random_warm_start:
                print("AL: RANDOM")
                new_ids = AL_random(take_until)
            elif "uncertainty-margin" in AL_parameters.AL_method:
                print("AL: UNCERTAINTY-MARGIN")
                new_ids = AL_uncertainty_margin(new_data, take_until, 
                                                model, trainer, 
                                                workers_available, 
                                                batch_size, 'diversity' in AL_parameters.AL_method)
            elif AL_parameters.AL_method == "diversity-cluster":
                print("AL: DIVERSITY-CLUSTER")
                new_ids = AL_diversity_cluster(get_graph_embeddings_mean(new_data), take_until)
            elif "deep-discriminator" in AL_parameters.AL_method:
                print("AL: deep-discriminator")
                if current_loaders['val'] is not None:
                    new_ids = AL_deep_discriminator(current_loaders['val'], 
                                                    new_data, take_until, 
                                                    model, 'diversity' in AL_parameters.AL_method)
                else:
                    new_ids = AL_random(take_until)
            elif "deep-adversarial" in AL_parameters.AL_method:
                print("AL: deep-adversarial")
                if current_loaders['val'] is not None:
                    new_ids = AL_deep_adversarial(current_loaders['train'], 
                                                  current_loaders['val'], 
                                                  new_data, take_until, model, 'diversity' in AL_parameters.AL_method)
                else:
                    new_ids = AL_random(take_until)
            elif AL_parameters.AL_method == "combined":
                new_ids_divided = []
                AL_nums_divided = []
                if AL_parameters.combined_AL_nums[0]>0:
                    new_ids_divided.append(AL_random(take_until))
                    AL_nums_divided.append(AL_parameters.combined_AL_nums[0])
                if AL_parameters.combined_AL_nums[1]>0:
                    new_ids_divided.append(AL_uncertainty_margin(new_data, take_until, model))
                    AL_nums_divided.append(AL_parameters.combined_AL_nums[1])
                if AL_parameters.combined_AL_nums[2]>0:
                    new_ids_divided.append(AL_diversity_cluster(new_data, take_until, AL_parameters.diversity_nums))
                    AL_nums_divided.append(AL_parameters.combined_AL_nums[2])
                
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

    print("NEW IDS:",new_ids)
    rem_data = None
    if new_ids is not None:
        #if AL_parameters.number_AL_iteration>0: #if offline
        #print(new_ids)
        #print(type(new_ids))
        rem_range = np.array(list(set(range(len(new_data))).difference(set(list(new_ids)))))
        #print(rem_range)
        rem_data = torch.utils.data.Subset(new_data, rem_range)
        
        new_data = torch.utils.data.Subset(new_data, new_ids)

    app = DataLoader(new_data, batch_size=len(new_data))
    cont = 0
    for dat in app:
        cont+=np.sum(dat.y.cpu().detach().numpy())
    new_positives = cont
    new_negatives = len(new_data) - new_positives

    new_train = new_data
    if current_data['train'] is None: #First batch
        current_data['train'] = new_data
    else:
        current_data['train'] = ConcatDataset([current_data['train'],new_train])      
        
        if AL_parameters.train_last_samples!=np.inf:
            current_data['train'] = current_data['train'][-AL_parameters.train_last_samples:]

    return current_data, rem_data, (new_positives, new_negatives)

def split_by_ratio(new_x, new_y, train_val_test_split_ratio):
    app = int(new_x.shape[0] * train_val_test_split_ratio[0])

    new_x_train = new_x[:app]
    new_y_train = new_y[:app]
    
    new_x_valid = new_x[app:]
    new_y_valid = new_y[app:]

    return new_x_train,new_y_train,new_x_valid,new_y_valid

def AL_random(take_until):
    app = np.array(range(take_until))
    return app

def AL_uncertainty_margin(new_x, take_until, model, trainer, workers_available, batch_size, use_diversity):
    model_input = DataLoader(new_x, batch_size=batch_size, num_workers=workers_available)

    predictions = trainer.predict(model, model_input)#[0]

    predictions = torch.cat(predictions, 0)

    pred_y = np.exp(predictions[:,1]).numpy()

    from_most_uncertain_ids = np.argsort(np.abs(pred_y-0.5))
    
    if use_diversity:
        most_uncertain_ids = from_most_uncertain_ids[:(take_until*10)]
        
        selected_ids = AL_diversity_cluster(get_graph_embeddings_mean(
            torch.utils.data.Subset(new_x, most_uncertain_ids)), take_until)
        selected_ids = most_uncertain_ids[selected_ids]
    else:
        selected_ids = from_most_uncertain_ids[:take_until]

    del model_input

    return selected_ids

def AL_diversity_cluster(x, take_until):
    num_clusters = take_until #at least 1

    cluster_result = KMeans(n_clusters = num_clusters, random_state=0).fit(x)

    cluster_centers = cluster_result.cluster_centers_

    centroids_ids = []
    for i in range(num_clusters):
        idx_cluster_i = np.where(cluster_result.labels_==i)[0]
        
        cos_sims = cosine_similarity(x[idx_cluster_i],cluster_centers[i:(i+1)])[:,0]
        sorted_idx = np.argsort(cos_sims)
    
        centroids_ids.append(sorted_idx[0])
        
    return centroids_ids

def AL_deep_discriminator(current_val_loader, new_x, take_until, model, use_diversity):
    discriminator_y, discriminator_x = model.get_output_and_embeddings(current_val_loader)#[0]

    discriminator_y = torch.Tensor(discriminator_y).long()
    discriminator_x = torch.Tensor(discriminator_x)

    embeddings_size = discriminator_x.shape[1]
    
    counts = torch.bincount(discriminator_y)

    loss_weights = max(counts)/counts

    loss_weights = loss_weights.to('cuda:{}'.format(model.cfg.gpus_available[0]))

    model.cfg.deep_al_weights = loss_weights

    discriminator_model = graph_model.MultiLabelClassifier(model.cfg, embeddings_size)
    
    #split this data
    train_x, val_x, train_y, val_y = train_test_split(discriminator_x, discriminator_y, test_size=0.2, random_state=42)

    train_data = DataLoader(TensorDataset(train_x,train_y),
                            batch_size=model.cfg.batch_size, 
                            shuffle=True, num_workers=model.cfg.workers_available, 
                            pin_memory=True)
    val_data = DataLoader(TensorDataset(val_x,val_y),
                          batch_size=model.cfg.batch_size, shuffle=False, 
                          num_workers=model.cfg.workers_available, pin_memory=True)

    es = pl.callbacks.EarlyStopping(monitor="validation_loss", patience=5) #validation_f1_score_macro / validation_loss
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="validation_loss", mode='min')

    trainer = pl.Trainer(
        gpus=model.cfg.gpus_available,
        max_epochs=model.cfg.epochs,
        accelerator="auto",
        #logger=wandb_logger,
        callbacks=[es, checkpointing],
        stochastic_weight_avg=True,
        accumulate_grad_batches=2,
        precision=16,
    )
    trainer.fit(discriminator_model, train_data, val_data)
    
    inference_dataloader = DataLoader(new_x, batch_size=model.cfg.batch_size, 
                                      shuffle=False, num_workers=model.cfg.workers_available, 
                                      pin_memory=True)

    _ , inference_x = model.get_output_and_embeddings(inference_dataloader)#[0]

    inference_dataloader = DataLoader(inference_x, batch_size=model.cfg.batch_size, 
                                      shuffle=False, num_workers=model.cfg.workers_available, 
                                      pin_memory=True)

    predictions = trainer.predict(discriminator_model, inference_dataloader)#[0]

    predictions = torch.cat(predictions, 0)

    predictions = np.exp(predictions[:, 1]).numpy()

    from_most_error_ids = np.argsort(-predictions)
    if use_diversity:
        most_error_ids = from_most_error_ids[:(take_until*10)]
        
        selected_ids = AL_diversity_cluster(get_graph_embeddings_mean(torch.utils.data.Subset(new_x, most_error_ids)), take_until)
        selected_ids = most_error_ids[selected_ids]
    else:
        selected_ids = from_most_error_ids[:take_until]

    return selected_ids

def AL_deep_adversarial(current_train_loader, current_val_loader, new_x, take_until, model, use_diversity):
    _, discriminator_x1 = model.get_output_and_embeddings(current_train_loader)#[0]
    discriminator_y1 = torch.zeros(discriminator_x1.shape[0]).long()
    discriminator_x1 = torch.Tensor(discriminator_x1)
    _, discriminator_x2 = model.get_output_and_embeddings(current_val_loader)#[0]
    discriminator_y2 = torch.ones(discriminator_x2.shape[0]).long()
    discriminator_x2 = torch.Tensor(discriminator_x2)

    discriminator_y = torch.cat((discriminator_y1,discriminator_y2))
    discriminator_x = torch.cat((discriminator_x1,discriminator_x2))

    embeddings_size = discriminator_x.shape[1]
    
    counts = torch.bincount(discriminator_y)

    loss_weights = max(counts)/counts

    loss_weights = loss_weights.to('cuda:{}'.format(model.cfg.gpus_available[0]))

    model.cfg.deep_al_weights = loss_weights

    discriminator_model = graph_model.MultiLabelClassifier(model.cfg, embeddings_size)
    
    #split this data
    train_x, val_x, train_y, val_y = train_test_split(discriminator_x, discriminator_y, 
                                                      test_size=0.2, random_state=42)

    train_data = DataLoader(TensorDataset(train_x,train_y),
                            batch_size=model.cfg.batch_size, shuffle=True, 
                            num_workers=model.cfg.workers_available, pin_memory=True)
    
    val_data = DataLoader(TensorDataset(val_x,val_y),
                          batch_size=model.cfg.batch_size, shuffle=False, 
                          num_workers=model.cfg.workers_available, pin_memory=True)

    es = pl.callbacks.EarlyStopping(monitor="validation_loss", patience=5) #validation_f1_score_macro / validation_loss
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="validation_loss", mode='min')

    trainer = pl.Trainer(
        gpus=model.cfg.gpus_available,
        max_epochs=model.cfg.epochs,
        accelerator="auto",
        #logger=wandb_logger,
        callbacks=[es, checkpointing],
        stochastic_weight_avg=True,
        accumulate_grad_batches=2,
        precision=16,
    )
    trainer.fit(discriminator_model, train_data, val_data)
    
    inference_dataloader = DataLoader(new_x, batch_size=model.cfg.batch_size, 
                                      shuffle=False, num_workers=model.cfg.workers_available, 
                                      pin_memory=True)

    _ , inference_x = model.get_output_and_embeddings(inference_dataloader)#[0]

    inference_dataloader = DataLoader(inference_x, batch_size=model.cfg.batch_size, 
                                      shuffle=False, num_workers=model.cfg.workers_available, 
                                      pin_memory=True)

    predictions = trainer.predict(discriminator_model, inference_dataloader)#[0]

    predictions = torch.cat(predictions, 0)

    predictions = np.exp(predictions[:, 1]).numpy()

    from_most_error_ids = np.argsort(-predictions)
    if use_diversity:
        most_error_ids = from_most_error_ids[:(take_until*10)]
        
        selected_ids = AL_diversity_cluster(get_graph_embeddings_mean(torch.utils.data.Subset(new_x, most_error_ids)), take_until)
        selected_ids = most_error_ids[selected_ids]
    else:
        selected_ids = from_most_error_ids[:take_until]

    return selected_ids


def get_graph_embeddings_mean(data):
    ls = []
    for dat in data:
        ls.append(dat.x.mean(dim=0))
    app = torch.stack(ls).cpu().detach().numpy()
    return app