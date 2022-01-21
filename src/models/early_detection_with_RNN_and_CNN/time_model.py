import sys, os
sys.path.append('..')

project_folder = os.path.join('../../../')

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Concatenate, GRU, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

conv_len = 3

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

    if len(class_weight)<2:
        print("!!!"*10,"CLASS WEIGHT ERROR:",class_weight,"!!!"*10)
        class_weight = {0:1, 1:1} 

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