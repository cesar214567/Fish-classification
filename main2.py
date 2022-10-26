import numpy as np
#np.random.seed(2016)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, log_loss
from keras import __version__ as keras_version

import tensorflow as tf

import torch                                          
import torchvision.models as models                   
from PIL import Image                                 
import torchvision.transforms.functional as TF        
from torchsummary import summary                       
import numpy as np
import torch.nn as nn
import torch.optim as optim

#import nvidia_smi

#print(torch.version.cuda)
#print(torch.cuda.is_available())
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                                                      
#columns=['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

#tf.debugging.set_log_device_placement(True)

columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone']

#print("device is: ",device)
#torch.cuda.set_per_process_memory_fraction(1.0, device=None) 
def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (244, 244), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    #folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in columns:
        index = columns.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        #path = os.path.join('.','NatureConservancy', 'train', fld, '*.jpg')
        path = os.path.join('.','FishSpecies', 'Training_Set', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            #print(fl)
            flbase = os.path.basename(fl)
            #img = get_im_cv2(fl)
            #print(img.shape)
            img = cv2.imread(fl)
            if img is None: 
                continue
            img = cv2.resize(img, (244, 244), cv2.INTER_LINEAR)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    #path = os.path.join('.','NatureConservancy',  'test_stg1', '*.jpg')
    path = os.path.join('.','FishSpecies', 'Training_Set', fld, '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=columns)
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    #print('Reshape...')
    #train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, len(columns))

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret



def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def create_model(): 
    IMG_SIZE = 224
    #tf.keras.applications.efficientnet.EfficientNetB0(include_top=True).summary()
    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(244,244,3),weights='imagenet',include_top=False )
    
    print(pretrained_model.output)
    layers = [1024,256,64]
    sequential_layers = [pretrained_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Flatten()]
    top_dropout_rate = 0.35 
    for layer in layers:
        sequential_layers.append(tf.keras.layers.BatchNormalization())
        sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
        sequential_layers.append(tf.keras.layers.Dense(layer, activation="relu"))    
    sequential_layers.append(tf.keras.layers.BatchNormalization())
    sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
    sequential_layers.append(tf.keras.layers.Dense(len(columns), activation="softmax", name="pred"))
    model = tf.keras.Sequential(sequential_layers)
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=[tf.keras.metrics.CategoricalAccuracy()],

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 128
    num_epoch = 80
    random_state = 51

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    best_model = 0
    best_model_error = 1000
    
    for train_index, test_index in kf.split(train_data):
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
               
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print("fold: ",num_fold)

        history = model.fit(
            X_train,
            Y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_valid, Y_valid),
        )
        exit()
        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(test_index,batch_size=32,shuffle=False,num_workers=4)
            Y_valid = torch.tensor(train_target[test_index])
            Y_valid = torch.argmax(Y_valid,dim=1)
            predictions = torch.tensor([]).cpu()
            
            for test_batch in test_loader:
                X_valid = train_data[test_batch]
                X_valid = torch.tensor(X_valid).to(device)
                predictions_valid = model(X_valid).cpu()
                predictions = torch.concat((predictions,predictions_valid))
            predictions_labels = torch.argmax(predictions,dim=1)
            conf_matrix = confusion_matrix(Y_valid.cpu(),predictions_labels.cpu())
            print(conf_matrix)
            print("aciertos: ",np.trace(conf_matrix))
            print("fallas: ",np.sum(conf_matrix))
        
            current_model_error = loss_fn(predictions,Y_valid.cpu()) 
            print("current error is: ",current_model_error)
            if (current_model_error < best_model_error):
                best_model_error = current_model_error
                best_model = model
            info_string = 'loss_' + str(best_model_error) + '_folds_' + str(nfolds)
            print(info_string)
    return info_string, best_model


def run_cross_validation_process_test(info_string, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 7
    info_string, model = run_cross_validation_create_models(num_folds)
    #run_cross_validation_process_test(info_string, models)
