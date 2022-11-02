import numpy as np
#np.random.seed(2016)

import ctypes
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudart64_110.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublas64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublasLt64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cufft64_10.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\curand64_10.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cusolver64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cusparse64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_train64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_cnn_train64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_train64_8.dll')

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, log_loss
from keras import __version__ as keras_version
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf

import numpy as np
from keras_dataloader.datagenerator import DataGenerator
                                                
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
columns=['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)
#print("device is: ",device)
#torch.cuda.set_per_process_memory_fraction(1.0, device=None) 
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
        path = os.path.join('.','NatureConservancy', 'train', fld, '*.jpg')
        #path = os.path.join('.','FishSpecies', 'Training_Set', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            #print(fl)
            flbase = os.path.basename(fl)
            #img = get_im_cv2(fl)
            #print(img.shape)
            img = cv2.imread(fl)
            if img is None: 
                continue
            img = cv2.resize(img, IMG_SIZE, cv2.INTER_LINEAR)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    X_test = []
    Y_test = []
    start_time = time.time()

    print('Read test images')
    for fld in columns:
        index = columns.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('.','NatureConservancy', 'train', fld, '*.jpg')
        #path = os.path.join('.','FishSpecies', 'Test_Set', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            #print(fl)
            flbase = os.path.basename(fl)
            #img = get_im_cv2(fl)
            #print(img.shape)
            img = cv2.imread(fl)
            if img is None: 
                continue
            img = cv2.resize(img, IMG_SIZE, cv2.INTER_LINEAR)
            X_test.append(img)
            Y_test.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, Y_test


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
    #train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, len(columns))

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    X_test = np.array(test_data, dtype=np.uint8)
    test_id = np.array(test_id, dtype=np.uint8)
    #create one hot encoding
    Y_test = np_utils.to_categorical(test_id, len(columns))

    #X_test = X_test.transpose((0, 3, 1, 2))

    X_test = X_test.astype('float32')
    #X_test = X_test / 255

    print('Test shape:', X_test.shape)
    print('Test shape:', test_id.shape)
    print(X_test.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, Y_test


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
    #tf.keras.applications.efficientnet.EfficientNetB0(include_top=True).summary()
    #pretrained_model = tf.keras.applications.VGG19(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.resnet.ResNet50(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(244,244,3),weights='imagenet',include_top=False )
    pretrained_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #for efficientnets pls check: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    #pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    
    
    #pretrained_model = tf.keras.applications.efficientnet.EfficientNetB2(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )

    pretrained_model.summary()
    
    pretrained_model.trainable = False

    print(pretrained_model.output)
    layers = [1024,256,64]
    sequential_layers = [pretrained_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Flatten()]
    #sequential_layers = [pretrained_model,tf.keras.layers.Flatten()]
    top_dropout_rate = 0.5 
    for layer in layers:
        sequential_layers.append(tf.keras.layers.BatchNormalization())
        sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
        sequential_layers.append(tf.keras.layers.Dense(layer, activation="relu"))    
    sequential_layers.append(tf.keras.layers.BatchNormalization())
    sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
    sequential_layers.append(tf.keras.layers.Dense(len(columns), activation="softmax", name="pred"))
    model = tf.keras.Sequential(sequential_layers)
    model.summary()
    
    optimizer = tf.keras.optimizers.RMSprop()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=['accuracy'],

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models():
    # input image dimensions
    num_epoch = 80
    X_train, Y_train, train_id = read_and_normalize_train_data()
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)
    
    model = create_model()
    
    train_generator = DataGenerator(X_train,Y_train, 4)
    validate_generator = DataGenerator(X_valid,Y_valid, 4)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(
        train_generator,
        epochs=num_epoch,
        validation_data=validate_generator,
        callbacks=[callback]
    )
    #test data 
    Y_test = Y_valid
    #X_test, Y_test = read_and_normalize_test_data()
    #validate_generator = DataGenerator(X_test, Y_test, 4)
    results = model.evaluate(validate_generator)
    predictions = model.predict(validate_generator)

    Y_test = np.argmax(Y_test,axis=1)
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(Y_test,predictions_labels)
    print(conf_matrix)
    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    run_cross_validation_create_models()
    #run_cross_validation_process_test(info_string, models)
