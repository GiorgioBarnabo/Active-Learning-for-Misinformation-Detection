import sys, os, json, time, datetime
sys.path.append('..')
import utils

project_folder = os.path.join('../../../')

import numpy as np
import random
import keras
import time
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, TimeDistributed, LSTM, AveragePooling1D, Embedding, GRU, GlobalAveragePooling1D
from keras import initializers, regularizers
from keras.initializers import RandomNormal
from keras.callbacks import CSVLogger, ModelCheckpoint

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

conv_len = 3
cut_off = 0.5

def rc_model(input_shape):
    print("input_shape[0]", input_shape[0])
    print("input_shape[1]", input_shape[1])
    input_f = Input(shape=(input_shape[0], input_shape[1], ),dtype='float32',name='input_f')
    r = GRU(64, return_sequences=True)(input_f)
    r = GlobalAveragePooling1D()(r)
    
    c = Conv1D(64, conv_len, activation='relu')(input_f)
    #c = Conv1D(64, conv_len, activation='relu')(c)
    c = MaxPooling1D(3)(c)
    c = GlobalAveragePooling1D()(c)

    rc = Concatenate()([r,c]) 
    rc = Dense(64, activation='relu')(rc)
    output_f = Dense(1, activation='sigmoid', name = 'output_f')(rc)
    model = Model(inputs=[input_f], outputs = [output_f])
    return model

def model_train(model, file_name, data, epochs, batch_size, iteration_num = 0):
    model.compile(loss={'output_f': 'binary_crossentropy'}, optimizer='rmsprop', metrics=['accuracy'])
    call_back = ModelCheckpoint(file_name, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    input_train = data['x_train']
    output_train = data['y_train']
    input_valid = data['x_valid']
    output_valid = data['y_valid']

    y_classes = np.unique(data['y_train'])
    class_weight = compute_class_weight('balanced',
                                        classes = y_classes,
                                        y = data['y_train'])
    class_weight = dict(zip(y_classes,class_weight))

    history = model.fit(input_train, output_train,
                        validation_data = (input_valid, output_valid),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[call_back], class_weight = class_weight,
                        verbose = 0)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train loss","val loss"])
    plt.savefig(os.path.join(file_name, '_loss_it_'+str(iteration_num)+'.png'))
    plt.close()

def model_predict(model, x):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    pred_test = pred_test.reshape((pred_test.shape[0],))
    return pred_test

def model_evaluate(model, x, y):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    rs = compute_metrics(pred_test, y_test)
    return rs

##NEW FUNCTIONS FEDERICO
def load_all_ordered(path, stop_at_year):
    all_loaded_data = {}
    all_eval_data = {}
    for filename in os.listdir(path):
        app = filename.split("_")
        if len(app)==1: #x.npy or others
            continue
        elif "eval" in filename and "-" in filename:
            is_eval = True
            _,file_year_month,is_y = app
        else:
            is_eval = False
            file_year_month,is_y = app
            
        if "-" in file_year_month: #to avoid evaluation_x/... files
            app = file_year_month.split("-")
            file_year = int(app[0])
            file_month = int(app[1])

            if file_year < stop_at_year:
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

    all_year_month_ordered_keys = sorted(all_loaded_data.keys())
    all_year_month_test_ordered_keys = sorted(all_eval_data.keys())

    '''
    if all_year_month_ordered_keys == all_year_month_test_ordered_keys:
        print("SAME MONTHS")
    else:
        print("!!!!!! ERROR - NOT SAME MONTHS!!!!!!!!") #### CAN HAPPEN IF TRAIN/TEST HAVE EMPTY MONTHS
    '''

    return all_year_month_ordered_keys, all_loaded_data, all_year_month_test_ordered_keys, all_eval_data

def get_current_key_id(all_year_month_ordered_keys, warm_start_year):
    for current_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>warm_start_year: #until warm_start_year INCLUDED
            break
    
    return current_key_id

def prepare_new_data_in_range(all_data, seq_len, data_opt, all_year_month_ordered_keys, starting_key_id, current_key_id):
    new_x, new_y = get_new_data_in_range(all_data, all_year_month_ordered_keys, starting_key_id, current_key_id)

    new_x = prepare_data(new_x, seq_len, data_opt)

    return new_x, new_y

def get_new_data_in_range(all_data, all_year_month_ordered_keys, starting_key_id, current_key_id):
    new_x = []
    new_y = []
    for key in all_year_month_ordered_keys[starting_key_id:current_key_id]:
        new_x.append(all_data[key][0])
        new_y.append(all_data[key][1])

    new_x = np.concatenate(new_x)
    new_y = np.concatenate(new_y)

    return new_x, new_y

def prepare_data(x,seq_len,data_opt):
    #print("shape data current experiment: ", x.shape) 
            
    x1 = x[:, 0:seq_len, :]
    
    #print("shape data current experiment: ", x1.shape) 

    shape = x1.shape
    x1 = x1.reshape([shape[0] * shape[1], shape[2]])
    #print("shape data current experiment: ", x1.shape)

    if 'twitter' in data_opt:
        pos_norm = [0,1,2,3,4,5,6,7,8]
    else:
        pos_norm = [0,1,2,3,4,5,6,7,8,9]

    pos_norm = [0,1,2,3,4,5,6,7,8,9,10,11,12]

    x1 = utils.normalize(x1, pos_norm)

    x1 = x1.reshape([shape[0], shape[1], shape[2]])

    #print("x1 shape, ", x1.shape)

    return x1

def prepare_test_data(all_test_data,seq_len,data_opt):
    prepared_test_data = {}
    for key,(x,y) in all_test_data.items():
        if x.shape[0]>1: #some tests are empty.....
            prepared_test_data[key] = [prepare_data(x,seq_len,data_opt),y]

    all_nonempty_year_month_test_ordered_keys = sorted(prepared_test_data.keys())

    return all_nonempty_year_month_test_ordered_keys, prepared_test_data

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

def merge_new_data(data, new_x, new_y, train_val_test_split_ratio, AL_method, model, num_urls_k, combined_AL_nums, diversity_nums, val_last_samples): #Active Learning Method
    if data['x_train'] is None:
        new_ids = None
    else:
        if new_x.shape[0]<=num_urls_k: #x
            new_ids = None
        else:
            take_until = min(new_x.shape[0],num_urls_k)
            if AL_method == "random":
                new_ids = AL_random(take_until)
            elif AL_method == "uncertainty-margin":
                new_ids = AL_uncertainty_margin(new_x, take_until, model)
            elif AL_method == "diversity-cluster":
                new_ids = AL_diversity_cluster(new_x, take_until, diversity_nums)
            elif AL_method == "combined":
                new_ids_divided = []
                AL_nums_divided = []
                if combined_AL_nums[0]>0:
                    new_ids_divided.append(AL_random(take_until))
                    AL_nums_divided.append(combined_AL_nums[0])
                if combined_AL_nums[1]>0:
                    new_ids_divided.append(AL_uncertainty_margin(new_x, take_until, model))
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

    if new_ids is not None:
        new_x,new_y = new_x[new_ids],new_y[new_ids]
    new_x_train,new_y_train,new_x_valid,new_y_valid = split_by_ratio(new_x, new_y, train_val_test_split_ratio)

    if data['x_train'] is None: #First batch
        data['x_train'] = new_x_train
        data['y_train'] = new_y_train
        
        data['x_valid'] = new_x_valid
        data['y_valid'] = new_y_valid

        #data['x_test'] = new_x_test
        #data['y_test'] = new_y_test
    else:
        data['x_train'] = np.concatenate([data['x_train'],new_x_train])
        data['y_train'] = np.concatenate([data['y_train'],new_y_train])
        
        data['x_valid'] = np.concatenate([data['x_valid'],new_x_valid])
        data['y_valid'] = np.concatenate([data['y_valid'],new_y_valid])

        if val_last_samples!=np.inf:
            app_x = data['x_valid'][:-val_last_samples]
            app_y = data['y_valid'][:-val_last_samples]

            data['x_train'] = np.concatenate([data['x_train'],app_x])
            data['y_train'] = np.concatenate([data['y_train'],app_y])
            
            data['x_valid'] = data['x_valid'][-val_last_samples:]
            data['y_valid'] = data['y_valid'][-val_last_samples:]
            
        #data['x_test'] = np.concatenate([data['x_test'],new_x_test])
        #data['y_test'] = np.concatenate([data['y_test'],new_y_test])

    return data

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

def AL_uncertainty_margin(new_x, take_until, model):
    model_input = {'input_f': new_x}
    pred_y = model.predict(model_input)[:,0]

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

def model_evaluate_per_month(model, all_nonempty_year_month_test_ordered_keys, test_data):
    rs = []
    for key in all_nonempty_year_month_test_ordered_keys: 
        x,y = test_data[key]
        input_test = {'input_f': x}
        pred_test = model.predict(input_test) >= cut_off
        y_test = y
        rs.append(np.array(compute_metrics(pred_test, y_test)))
    return np.array(rs)

def compute_metrics(pred_test, y_test):
    (pre_0,pre_1),(rec_0,rec_1),(f_0,f_1),(_,_) = metrics.precision_recall_fscore_support(y_test,pred_test, labels=[0,1],zero_division=0)
    acc = metrics.accuracy_score(y_test,pred_test)
    #res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)
    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0]

    return res

################################################################################
####################################  MAIN  ####################################
################################################################################
def main():
    epochs = 50 #50
    batch_size = 128
    nb_sample = 1
    seq_lens = [10, 20, 40, 60, 80]
    data_opt =  'condor' #'twitter' #condor_gossipcop_politifact #condor #gossipcop #politifact

    warm_start_year = 2016 #to load all data until this year
    stop_at_year = 2020 #load until this year (NOT INCLUDED)

    AL_methods = ["random", "uncertainty-margin", "diversity-cluster", "combined"] #don't use "_" to keep filename easily separable
    #"random"
    #"uncertainty-margin"
    #"diversity-cluster"
    #"combined"

    train_val_test_split_ratio = [0.80, 0.20, 0.] #test is 0 because separated
    num_urls_k_list = [10, 30, 60, 90, 120]
    diversity_nums = [1,1,1] #centroids, outliers, randoms; used only if AL_method = diversity-cluster
    combined_AL_nums = [1,7,2] #ignored if AL_method is not "combined"
    #random, uncertainty, diversity
    val_last_samples_list = [3*k for k in num_urls_k_list] #Use np.inf to not discard validation (discard=add to train)
    retrain_from_scratch = False

    results_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'results', data_opt)
    #create results_folder if not exists
    if not os.path.isdir(results_folder):
      os.makedirs(results_folder)

    all_year_month_ordered_keys, all_data, all_year_month_test_ordered_keys, all_test_data = load_all_ordered(os.path.join(project_folder, 'data', 'features', data_opt), stop_at_year) #to load all data

    print(all_year_month_ordered_keys)

    nb_feature = all_data[all_year_month_ordered_keys[0]][0].shape[2]

    #current_key_id = current year-month id
    warm_start_key_id = get_current_key_id(all_year_month_ordered_keys, warm_start_year)
    
    for seq_len in seq_lens:
        print('seq_len {}'.format(seq_len))
        all_nonempty_year_month_test_ordered_keys, prepared_test_data = prepare_test_data(all_test_data,seq_len,data_opt)

        for AL_method in AL_methods:
            print('AL_method {}'.format(AL_method))
            for num_urls_k,val_last_samples in zip(num_urls_k_list, val_last_samples_list):

                all_rs = np.zeros((nb_sample, len(all_year_month_ordered_keys)-warm_start_key_id +1,
                                   len(all_nonempty_year_month_test_ordered_keys),
                                   7,))
                for sample in range(nb_sample):
                    print('sample {}'.format(sample))

                    data = {}
                    
                    data['x_train'] = None
                    data['y_train'] = None
                    
                    data['x_valid'] = None
                    data['y_valid'] = None

                    #data['x_test'] = prepare_data(np.load(os.path.join(project_folder, 'data', 'features', data_opt,"evaluation_x.npy")), seq_len, data_opt)
                    #data['y_test'] = np.load(os.path.join(project_folder, 'data', 'features', data_opt,"evaluation_y.npy"))

                    model = None

                    starting_key_id = 0

                    #print(seq_len)
                    #print(nb_feature)

                    model = rc_model(input_shape = [seq_len, nb_feature])
                    #print(model.summary())

                    for iteration_num,current_key_id in enumerate(range(warm_start_key_id,len(all_year_month_ordered_keys)+1)):
                        print("CURRENT PERIOD:", all_year_month_ordered_keys[starting_key_id], "(incl.) - ", all_year_month_ordered_keys[current_key_id-1],"(incl.)")
                        new_x, new_y = prepare_new_data_in_range(all_data, seq_len, data_opt, all_year_month_ordered_keys, starting_key_id, current_key_id)

                        print(new_x.shape, new_y.shape)

                        data = merge_new_data(data, new_x, new_y, train_val_test_split_ratio, AL_method, model, num_urls_k, combined_AL_nums, diversity_nums, val_last_samples)

                        print("data['x_valid'].shape, ", data['x_valid'].shape)
                        print("data['x_train'].shape, ", data['x_train'].shape)
                        #print("data['x_test'].shape, ", data['x_test'].shape)

                        model_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'rc_model', data_opt)
                        if not os.path.isdir(model_folder):
                            os.makedirs(model_folder)
                        #print("model folder name: ", model_folder)
                        model_name = 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k)
                        #print("model name: ", model_name)

                        if retrain_from_scratch:
                            model = rc_model(input_shape = [seq_len, nb_feature])
                        model_train(model, file_name = os.path.join(model_folder, model_name), data = data, epochs = epochs, batch_size = batch_size, iteration_num = iteration_num)

                        model = load_model(os.path.join(model_folder, model_name))

                        rs = model_evaluate_per_month(model, all_nonempty_year_month_test_ordered_keys, prepared_test_data)
                    
                        all_rs[sample,iteration_num,:,:] += rs

                        starting_key_id = current_key_id
                
                with open(os.path.join(results_folder, 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k) + '.npy'), 'wb') as f:
                    np.save(f, all_rs)
                
if __name__ == '__main__':
    main()
    
