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
        rec_1 = tp_1 / (tp_1 + fn_1)
        f_1 = 2 * tp_1 / (2 * tp_1 + fp_1 + fn_1)
        
        pre_0 = tp_0 / (tp_0 + fp_0)
        rec_0 = tp_0 / (tp_0 + fn_0)
        f_0 = 2 * tp_0 / (2 * tp_0 + fp_0 + fn_0)
    except:
        return None
    
    #res = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0)
    res = [acc, pre_1, rec_1, f_1, pre_0, rec_0, f_0]
    
    return res

def model_evaluate(model, x, y):
    input_test = {'input_f': x}
    pred_test = model.predict(input_test)
    y_test = y
    rs = compute_metrics(pred_test, y_test)
    return rs

def main():
    #seq_len = 35
    epochs = 100
    batch_size = 128
    nb_sample = 1
    seq_lens = [5, 10, 20, 40, 60, 80]
    data_opt =  'condor' #'twitter'
    
    if data_opt =='twitter':
        data_name = 'twitter15'
    else:
        data_name = 'weibo'
    
    x = np.load(os.path.join(project_folder, 'data', 'features', data_opt, 'x.npy'))
    y = np.load(os.path.join(project_folder, 'data', 'features', data_opt, 'y.npy'))
    print("data shapes: ", x.shape, y.shape)
    
    n = x.shape[0]
    nb_feature = x.shape[2]
    x = x.astype('float32')
    pos = np.arange(n)

    print("total number of rows", n)
    
    time.sleep()

    rs_avg = {}
    for seq_len in seq_lens:
        rs_avg[seq_len] = [0 for i in range(7)]
    
    split_ratio = [0.01, 0.39, 0.60]
    
    for sample in range(nb_sample):

        print('sample {}'.format(sample))
        
        #np.random.shuffle(pos[0:21855]) # We only shuffle urls before the test set
        np.random.shuffle(pos)

        for seq_len in seq_lens:

            print("shape data current experiment: ", x.shape) 
            
            x1 = x[:, 0:seq_len, :]
            
            print("shape data current experiment: ", x1.shape) 

            shape = x1.shape
            x1 = x1.reshape([shape[0] * shape[1], shape[2]])
            print("shape data current experiment: ", x1.shape)

            if 'twitter' in data_opt:
                pos_norm = [0,1,2,3,4,5,6,7,8]
            else:
                pos_norm = [0,1,2,3,4,5,6,7,8,9]

            pos_norm = [0,1,2,3,4,5,6,7,8,9,10,11,12]

            x1 = utils.normalize(x1, pos_norm)

            x1 = x1.reshape([shape[0], shape[1], shape[2]])

            print("x1 shape, ", x1.shape)

            data = {}

            data['x_train'] = x1[pos[0:int(n * split_ratio[0])], :]
            data['y_train'] = y[pos[0:int(n * split_ratio[0])]]
            
            data['x_valid'] = x1[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])], :]
            data['y_valid'] = y[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])]]

            # data['x_valid'] = x1[pos[0:int(n * split_ratio[0])], :]
            # data['y_valid'] = y[pos[0:int(n * split_ratio[0])]]
            
            # data['x_train'] = x1[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])], :]
            # data['y_train'] = y[pos[int(n * split_ratio[0]): int(n * split_ratio[0]) + int(n * split_ratio[1])]]

            data['x_test'] = x1[pos[int(n * split_ratio[0]) + int(n * split_ratio[1]): ], :]
            data['y_test'] = y[pos[int(n * split_ratio[0]) + int(n * split_ratio[1]): ]]

            print("data['x_valid'].shape, ", data['x_valid'].shape)
            print("data['x_train'].shape, ", data['x_train'].shape)
            print("data['x_test'].shape, ", data['x_test'].shape)

            model = rc_model(input_shape = [seq_len, nb_feature])
            print(model.summary())

            
            model_folder = os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'rc_model')
            print("model folder name: ", model_folder)
            model_name = 'sp_{}_seqlen_{}'.format(sample, seq_len)
            print("model name: ", model_name)
            model_train(model, file_name = os.path.join(model_folder, model_name), data = data, epochs = epochs, batch_size = batch_size)
            
            # print("success")
            # time.sleep(1000)
            
            best_model = load_model(os.path.join(model_folder, model_name))

            rs = model_evaluate(best_model, x = data['x_test'], y = data['y_test'])
            f = open(os.path.join(project_folder, 'src', 'models', 'early_detection_with_RNN_and_CNN', 'results', 'current_results.txt'), 'a')
            info = 'sp_{}_seqlen_{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(sample, seq_len, rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6])
            f.write(info)
            f.close()
            print(info)
        
            for i in range(7):
                rs_avg[seq_len][i] += rs[i]
    
    for seq_len in seq_lens:
        for i in range(7):
            rs_avg[seq_len][i] /= nb_sample
        rs = rs_avg[seq_len]
        print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6]))

                
if __name__ == '__main__':
    main()
    
