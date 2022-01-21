import sys, os, argparse
sys.path.append('..')

project_folder = os.path.join('../../../')

import numpy as np
from keras.models import load_model

import tensorflow as tf

'''
sys.path.insert(1, os.getcwd()+'/utils')
from data_loader import *
from eval_helper import *
'''

import torch
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataLoader, DataListLoader

#import our scripts
import AL
import data_utils
import graph_model
import time_model

################################################################################
####################################  MAIN  ####################################
################################################################################
def main():
    nb_sample = 1
    results_set = 'RNN_GNN' #specify characteristics current set of experiments
    ''' Results names
    GAT
    GCN
    RNN_GNN
    '''
    #data parameters
    dataname =  'gossipcop' #'twitter' #condor_gossipcop_politifact #condor #gossipcop #politifact
    warm_start_years = [np.inf,np.inf] #warm-start data from year[0] (included) to year[1] (included)
                                   #to avoid warm-start: pick np.inf
    training_years = [2005,2021] #to train (after warm-start) from year[0] (included) to year[1] (included)
    
    #model parameters
    batch_size = 128 #128 conv #128 graph_model
    model_type = "time" #time #graph

    train_val_test_split_ratio = [0.80, 0.20, 0.] #test is 0 because separated
    if model_type=="time":#time_model parameters
        epochs = 50 #50 time_model
        seq_lens = [80]

        graph_test_size = None
        graph_feature = None
    elif model_type=="graph": #graph_model parameters
        epochs = 35 #35 graph_model
        graph_test_size = 0.25 #because is not divided --> change if split data
        graph_feature = "bert" #profile, spacy, bert, content
        graph_model_name = "gcn" #gcn, gat, sage
        seq_lens = [np.inf]

        
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
        parser.add_argument('--nhid', type=int, default=128, help='hidden size')
        parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
        parser.add_argument('--epochs', type=int, default=epochs, help='maximum number of epochs')
        parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
        parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
        parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
        parser.add_argument('--feature', type=str, default=graph_feature, help='feature type, [profile, spacy, bert, content]')
        parser.add_argument('--model', type=str, default=graph_model_name, help='model type, [gcn, gat, sage]')
        graph_args = parser.parse_args()

        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)

    #Active Learning parameters
    offline_AL = 60 #0 == Online, >0 == Number of iterations for Offline AL
    AL_methods = ["uncertainty-margin", "random", "diversity-cluster", "combined"] #["combined", "uncertainty-margin", "diversity-cluster", "random"] #don't use "_" to keep filename easily separable
    num_urls_k_list = [20]
    diversity_nums = [1,1,1] #centroids, outliers, randoms; used only if AL_method = diversity-cluster
    combined_AL_nums = [1,7,2] #ignored if AL_method is not "combined" #random, uncertainty, diversity

    train_last_samples_list = [np.inf for k in num_urls_k_list] #Use np.inf to not discard training 
    val_last_samples_list = [np.inf for k in num_urls_k_list] #Use np.inf to not discard validation
    add_val_to_train = False #(discard=add to train)
    retrain_from_scratch = True

    #to debug
    debugging = False #True
    if debugging: 
        epochs = 1
        seq_lens = [50]
        AL_methods = ["combined"]
        num_urls_k_list = [10]
        offline_AL = 36

    results_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', "results", results_set, dataname)
    #create results_folder not exists
    if not os.path.isdir(results_folder):
      os.makedirs(results_folder)

    data_folder = os.path.join(project_folder, 'data', ['features','graph'][model_type=="graph"], dataname)
    all_year_month_ordered_keys, all_data, all_year_month_test_ordered_keys, all_test_data, graph_dataset = data_utils.load_all_ordered(data_folder,
                                                                                                                         warm_start_years, training_years,
                                                                                                                         model_type,
                                                                                                                         graph_test_size, graph_feature) #to load all data

    print(all_year_month_ordered_keys)
    print(all_year_month_test_ordered_keys)

    if model_type=="time":#time_model parameters
        nb_feature = all_data[all_year_month_ordered_keys[0]][0].shape[2]
    elif model_type=="graph":
        graph_args.num_features = graph_dataset.num_features
        graph_args.num_classes = graph_dataset.num_classes

    #current_key_id = current year-month id
    warm_start_starting_key_id,warm_start_ending_key_id = data_utils.get_warm_start_key_ids(all_year_month_ordered_keys, warm_start_years)
    print(warm_start_starting_key_id,warm_start_ending_key_id)

    doing_warm_start = True
    if warm_start_starting_key_id==len(all_year_month_ordered_keys)-1 or warm_start_ending_key_id==0:
        print("!!! NOT DOING WARM START !!!")
        warm_start_starting_key_id = 0
        if offline_AL>0:
            warm_start_ending_key_id = len(all_year_month_ordered_keys)
        else:
            warm_start_ending_key_id = 1
        doing_warm_start = False

    for seq_len in seq_lens:
        print('seq_len {}'.format(seq_len))
        if model_type=="time":#time_model parameters
            all_nonempty_year_month_test_ordered_keys, prepared_test_data = data_utils.prepare_test_data(all_test_data,seq_len,dataname)
        elif model_type=="graph":
            all_nonempty_year_month_test_ordered_keys, prepared_test_data = all_year_month_test_ordered_keys, all_test_data
        
        for AL_method in AL_methods:
            print('AL_method {}'.format(AL_method))
            for enum_num_urls_k,num_urls_k in enumerate(num_urls_k_list):
                results_filename = os.path.join(results_folder, 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k) + '.npy')
                if not os.path.isfile(results_filename):
                    if offline_AL>0:
                        all_rs = np.zeros((nb_sample,
                                        offline_AL + 1*doing_warm_start,
                                        len(all_nonempty_year_month_test_ordered_keys),
                                        8,))
                        all_true_false_nums = np.zeros((nb_sample,
                                                        offline_AL + 1*doing_warm_start,
                                                        4,))
                    else:
                        all_rs = np.zeros((nb_sample,
                                        len(all_year_month_ordered_keys) - warm_start_ending_key_id +1,
                                        len(all_nonempty_year_month_test_ordered_keys),
                                        8,))

                        all_true_false_nums = np.zeros((nb_sample,
                                                        len(all_year_month_ordered_keys) - warm_start_ending_key_id +1,
                                                        4,))
                    
                    for sample in range(nb_sample):
                        print('sample {}'.format(sample))

                        data = {}
                        
                        data['x_train'] = None
                        data['y_train'] = None
                        
                        data['x_valid'] = None
                        data['y_valid'] = None

                        model = None

                        if model_type=="time":
                            tf.compat.v1.set_random_seed(123)
                            model = time_model.rc_model(input_shape = [seq_len, nb_feature])
                            #print(model.summary())
                        elif model_type=="graph":
                            model = graph_model.Model(graph_args, concat=graph_args.concat)
                            if graph_args.multi_gpu:
                                model = DataParallel(model)
                            model = model.to(graph_args.device)

                        if offline_AL>0:
                            if doing_warm_start:
                                iteration_ranges = list(zip([warm_start_starting_key_id]+[warm_start_ending_key_id]*offline_AL,
                                                            [warm_start_ending_key_id]+[len(all_year_month_ordered_keys)]*offline_AL))
                            else:
                                iteration_ranges = list(zip([warm_start_starting_key_id]*offline_AL,
                                                            [len(all_year_month_ordered_keys)]*offline_AL))
                        else:
                            iteration_ranges = list(zip([warm_start_starting_key_id]+list(range(warm_start_ending_key_id, len(all_year_month_ordered_keys))),
                                                   range(warm_start_ending_key_id, len(all_year_month_ordered_keys)+1)))

                        print("IT_RNG:",iteration_ranges)

                        done_keys = (None,None)
                        for iteration_num,(starting_key_id,current_key_id) in enumerate(iteration_ranges):
                            print("CURRENT PERIOD:", all_year_month_ordered_keys[starting_key_id], "(incl.) - ", all_year_month_ordered_keys[current_key_id-1],"(incl.)")
                            if offline_AL==0 or ((starting_key_id,current_key_id) != done_keys):
                                print("GETTING NEW DATA")
                                new_x, new_y = data_utils.prepare_new_data_in_range(all_data, seq_len, dataname,
                                                                                    all_year_month_ordered_keys, starting_key_id, current_key_id,
                                                                                    model_type)
                            else:
                                print("DATA ALREADY ACQUIRED")
                                if new_x is None:
                                    print("!!!!!"*10,"BREAK")
                                    break

                            #print(new_x.shape, new_y.shape)

                            print("NEW TRAINING DATA")

                            data, (new_x, new_y), (new_positives, new_negatives) = AL.merge_new_data(data, new_x, new_y, train_val_test_split_ratio,
                                                AL_method, doing_warm_start and iteration_num==0, offline_AL, model, num_urls_k, combined_AL_nums, diversity_nums,
                                                train_last_samples_list[enum_num_urls_k], val_last_samples_list[enum_num_urls_k], add_val_to_train, model_type)

                            print("FINISH GETTING NEW TRAINING DATA")

                            #save new_positives/negatives and current_positives/negatives
                            if model_type=="time":
                                current_positives = np.sum(data['y_train']) + np.sum(data['y_valid'])
                                current_negatives = len(data['y_train']) - current_positives
                            elif model_type=="graph":
                                if model.multi_gpu:
                                    loader = DataListLoader
                                else:
                                    loader = DataLoader
                                app = loader(data['x_train'], batch_size=model.batch_size)
                                cont = 0
                                for dat in app:
                                    cont+=np.sum(dat.y.cpu().detach().numpy())
                                current_positives = cont
                                current_negatives = len(data['x_train']) - new_positives

                            all_true_false_nums[sample,iteration_num,:] += np.array([new_positives, new_negatives,
                                                                                    current_positives,current_negatives])

                            print("data['x_train'].shape, ", len(data['x_train']))
                            print("data['x_valid'].shape, ", len(data['x_valid']))
                            
                            if new_x is not None:
                                print("REM SHAPES",len(new_x))#, len(new_y))

                            model_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', ['rc_model','graph_model'][model_type=="graph"], results_set, dataname)
                            if not os.path.isdir(model_folder):
                                os.makedirs(model_folder)
                            #print("model folder name: ", model_folder)
                            model_name = 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k)
                            #print("model name: ", model_name)

                            if retrain_from_scratch:
                                if model_type=="time":
                                    tf.compat.v1.set_random_seed(123)
                                    model = time_model.rc_model(input_shape = [seq_len, nb_feature])
                                    #print(model.summary())
                                elif model_type=="graph":
                                    model = graph_model.Model(graph_args, concat=graph_args.concat)
                                    if graph_args.multi_gpu:
                                        model = DataParallel(model)
                                    model = model.to(graph_args.device)

                            if model_type=="time":
                                time_model.model_train(model, file_name = os.path.join(model_folder, model_name),
                                                       data = data, epochs = epochs, batch_size = batch_size,
                                                       iteration_num = iteration_num)
                                model = load_model(os.path.join(model_folder, model_name))
                            elif model_type=="graph":
                                optimizer = torch.optim.Adam(model.parameters(), lr=graph_args.lr, weight_decay=graph_args.weight_decay)

                                if model.multi_gpu:
                                    loader = DataListLoader
                                else:
                                    loader = DataLoader
                                train_loader = loader(data["x_train"], batch_size=model.batch_size)
                                val_loader = loader(data["x_valid"], batch_size=model.batch_size)

                                model = model.training_loop(train_loader, val_loader, optimizer)
                            
                            rs = data_utils.model_evaluate_per_month(model, all_nonempty_year_month_test_ordered_keys, prepared_test_data, model_type)
                        
                            all_rs[sample,iteration_num,:,:] += rs

                            done_keys = starting_key_id,current_key_id
                    
                    with open(results_filename, 'wb') as f:
                        np.save(f, all_rs)

                    with open(os.path.join(results_folder, 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k) + '_sample_size.npy'), 'wb') as f:
                        np.save(f, all_true_false_nums)
                else:
                    print("!!!"*10, results_filename,"already exists --> SKIP", "!!!"*10)
                    
if __name__ == '__main__':
    main()
    
