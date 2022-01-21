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

def model_train(model, file_name, data, epochs, batch_size):
    model.compile(loss={'output_f': 'binary_crossentropy'}, optimizer='rmsprop',metrics=['accuracy'])
    call_back = ModelCheckpoint(file_name, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    input_train = {'input_f': data['x_train']}
    output_train = {'output_f': data['y_train']}
    input_valid = {'input_f': data['x_valid']}
    output_valid = {'output_f': data['y_valid']}

    model.fit(input_train, output_train, epochs=epochs, batch_size=batch_size, validation_data = (input_valid, output_valid), callbacks=[call_back], verbose = 0)

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

                #SHUFFLE DATA
                pos = np.arange(new_load.shape[0])
                np.random.shuffle(pos)
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

    if all_year_month_ordered_keys == all_year_month_test_ordered_keys:
        print("SAME MONTHS")
    else:
        print("!!!!!! ERROR - NOT SAME MONTHS!!!!!!!!")
    
    return all_year_month_ordered_keys, all_loaded_data, all_year_month_test_ordered_keys, all_eval_data

def get_current_key_id(all_year_month_ordered_keys, cold_start_year):
    for current_key_id, key in enumerate(all_year_month_ordered_keys):
        if int(key.split("-")[0])>cold_start_year: #until cold_start_year INCLUDED
            break
    
    return current_key_id

def prepare_new_data_in_range(current_x, current_y, all_data, seq_len, data_opt, all_year_month_ordered_keys, starting_key_id, current_key_id):
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

def merge_new_data(data, new_x, new_y, split_ratio, AL_method, model, num_urls_k): #Active Learning Method
    if data['x_train'] is None:
        pass
    elif AL_method == "random":
        new_x, new_y = ALrandom(new_x, new_y, num_urls_k)
    elif AL_method == "uncertainty-margin":
        new_x, new_y = AL_uncertainty_margin(new_x, new_y, model, num_urls_k)
    else:
        print("NOT IMPLEMENTED YET")

    new_x_train,new_y_train,new_x_valid,new_y_valid = split_by_ratio(new_x, new_y, split_ratio)

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

        #data['x_test'] = np.concatenate([data['x_test'],new_x_test])
        #data['y_test'] = np.concatenate([data['y_test'],new_y_test])

    return data

def split_by_ratio(new_x, new_y, split_ratio):
    new_x_train = new_x[0:int(new_x.shape[0] * split_ratio[0]), :]
    new_y_train = new_y[0:int(new_x.shape[0] * split_ratio[0])]
    
    new_x_valid = new_x[int(new_x.shape[0] * split_ratio[0]):, :]
    new_y_valid = new_y[int(new_x.shape[0] * split_ratio[0]):]

    #new_x_valid = new_x[int(new_x.shape[0] * split_ratio[0]): int(new_x.shape[0] * split_ratio[0]) + int(new_x.shape[0] * split_ratio[1]), :]
    #new_y_valid = new_y[int(new_x.shape[0] * split_ratio[0]): int(new_x.shape[0] * split_ratio[0]) + int(new_x.shape[0] * split_ratio[1])]

    #new_x_test = new_x[int(n * split_ratio[0]) + int(n * split_ratio[1]):, :]
    #new_y_test = new_y[int(n * split_ratio[0]) + int(n * split_ratio[1]):]

    return new_x_train,new_y_train,new_x_valid,new_y_valid #,new_x_test,new_y_test

def ALrandom(new_x, new_y, num_urls_k):
    take_until = min(new_x.shape[0],num_urls_k)

    return new_x[:take_until], new_y[:take_until]

def AL_uncertainty_margin(new_x, new_y, model, num_urls_k):
    model_input = {'input_f': new_x}
    pred_y = model.predict(model_input)[:,0]

    take_until = min(new_x.shape[0], num_urls_k)

    from_most_uncertain_ids = np.argsort(np.abs(pred_y-0.5))
    most_uncertain_ids = from_most_uncertain_ids[:take_until]

    return new_x[most_uncertain_ids], new_y[most_uncertain_ids]

def model_evaluate_per_month(model, all_nonempty_year_month_test_ordered_keys, test_data):
    rs = []
    for key in all_nonempty_year_month_test_ordered_keys: 
        x,y = test_data[key]
        input_test = {'input_f': x}
        pred_test = model.predict(input_test)
        y_test = y
        rs.append(np.array(compute_metrics(pred_test, y_test)))
    return np.array(rs)

def compute_metrics(pred_test, y_test):
    tp_1, tn_1, fp_1, fn_1, tp_0, tn_0, fp_0, fn_0 = 0, 0, 0, 0, 0, 0, 0, 0
    
    for i in range(pred_test.shape[0]):
        lp = pred_test[i]
        lt = y_test[i]
        if lp >= cut_off:
            lp = 1
        else:
            lp = 0
        if lp == 1 and lt == 1:
            tp_1 += 1
            tn_0 += 1
        if lp == 0 and lt == 0:
            tn_1 += 1
            tp_0 += 1
        if lp == 1 and lt == 0:
            fp_1 += 1
            fn_0 += 1
        if lp == 0 and lt == 1:
            fn_1 += 1
            fp_0 += 1
        
    acc = (tp_1 + tn_1) / (tp_1 + tn_1 + fp_1 + fn_1)
    acc_0 = (tp_0 + tn_0) / (tp_0 + tn_0 + fp_0 + fn_0)
    if acc != acc_0:
        print('error')
    
    try:
        pre_1 = tp_1 / (tp_1 + fp_1)
    except ZeroDivisionError:
        pre_1 = np.inf
    
    try:
        rec_1 = tp_1 / (tp_1 + fn_1)
    except ZeroDivisionError:
        rec_1 = np.inf

    try:
        f_1 = 2 * tp_1 / (2 * tp_1 + fp_1 + fn_1)
    except ZeroDivisionError:
        f_1 = np.inf

    try:
        pre_0 = tp_0 / (tp_0 + fp_0)
    except ZeroDivisionError:
        pre_0 = np.inf

    try:
        rec_0 = tp_0 / (tp_0 + fn_0)
    except ZeroDivisionError:
        rec_0 = np.inf

    try:
        f_0 = 2 * tp_0 / (2 * tp_0 + fp_0 + fn_0)
    except ZeroDivisionError:
        f_0 = np.inf
    
    #res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)
    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0]
    
    return res


####MAIN
def main():
    epochs = 100
    batch_size = 128
    nb_sample = 1
    seq_lens = [5, 10, 20, 40, 60, 80]
    data_opt =  'condor_gossipcop_politifact' #'twitter'

    cold_start_year = 2016 #to load all data until this year
    stop_at_year = 2020 #load until this year (NOT INCLUDED)

    AL_methods = ["uncertainty-margin","random"] #don't use "_" to keep filename easily separable
    
    if data_opt =='twitter':
        data_name = 'twitter15'
    else:
        data_name = 'weibo'
    
    all_year_month_ordered_keys, all_data, all_year_month_test_ordered_keys, all_test_data = load_all_ordered(os.path.join(project_folder, 'data', 'features', data_opt), stop_at_year) #to load all data

    print(all_year_month_ordered_keys)

    #current_key_id = current year-month id
    cold_start_key_id = get_current_key_id(all_year_month_ordered_keys, cold_start_year)

    current_x = None
    current_y = None

    split_ratio = [0.80, 0.20, 0.] #test is 0 because separated
    num_urls_k_list = [5, 10, 20, 30]
    
    for seq_len in seq_lens:
        print('seq_len {}'.format(seq_len))
        all_nonempty_year_month_test_ordered_keys, prepared_test_data = prepare_test_data(all_test_data,seq_len,data_opt)

        for AL_method in AL_methods:
            print('AL_method {}'.format(AL_method))
            for num_urls_k in num_urls_k_list:

                all_rs = np.zeros((nb_sample, len(all_year_month_ordered_keys)-cold_start_key_id,
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
                    for iteration_num,current_key_id in enumerate(range(cold_start_key_id,len(all_year_month_ordered_keys))):
                        print("CURRENT MONTH:",all_year_month_ordered_keys[current_key_id])
                        new_x, new_y = prepare_new_data_in_range(current_x, current_y, all_data, seq_len, data_opt, all_year_month_ordered_keys, starting_key_id, current_key_id)

                        data = merge_new_data(data, new_x, new_y, split_ratio, AL_method, model, num_urls_k)

                        print("data['x_valid'].shape, ", data['x_valid'].shape)
                        print("data['x_train'].shape, ", data['x_train'].shape)
                        #print("data['x_test'].shape, ", data['x_test'].shape)

                        nb_feature = data['x_train'].shape[2]

                        #print(seq_len)
                        #print(nb_feature)

                        model = rc_model(input_shape = [seq_len, nb_feature])
                        #print(model.summary())

                        model_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'rc_model')
                        #print("model folder name: ", model_folder)
                        model_name = 'sp_{}_seqlen_{}'.format(sample, seq_len)
                        #print("model name: ", model_name)
                        model_train(model, file_name = os.path.join(model_folder, model_name), data = data, epochs = epochs, batch_size = batch_size)

                        best_model = load_model(os.path.join(model_folder, model_name))

                        rs = model_evaluate_per_month(best_model, all_nonempty_year_month_test_ordered_keys, prepared_test_data)
                    
                        all_rs[sample,iteration_num,:,:] += rs

                        starting_key_id = current_key_id
                
                with open(os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'results', 'seqlen_'+str(seq_len) + '_ALm_'+AL_method + '_k_'+str(num_urls_k) + '.npy'), 'wb') as f:
                    np.save(f, all_rs)
                
if __name__ == '__main__':
    main()
    
