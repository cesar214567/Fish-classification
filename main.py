import numpy as np
np.random.seed(2022)
seed = 2022
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
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras_dataloader.datagenerator import DataGenerator
                                                
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
columns=['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone','Trout']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)
#print("device is: ",device)
#torch.cuda.set_per_process_memory_fraction(1.0, device=None) 
def load_train(folder):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    #folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in columns:
        index = columns.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('.',folder, 'train', fld, '*.jpg')
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


def load_test(folder):
    X_test = []
    Y_test = []
    start_time = time.time()

    print('Read test images')
    for fld in columns:
        index = columns.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('.',folder, 'test', fld, '*.jpg')
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


def read_and_normalize_train_data(folder):
    train_data, train_target, train_id = load_train(folder)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)
    #print('Reshape...')
    #train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    #train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, len(columns))

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data(folder):
    start_time = time.time()
    test_data, test_id = load_test(folder)

    X_test = np.array(test_data, dtype=np.float32)
    test_id = np.array(test_id, dtype=np.uint8)
    #create one hot encoding
    Y_test = np_utils.to_categorical(test_id, len(columns))

    #X_test = X_test.transpose((0, 3, 1, 2))

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

def get_vgg19(): 
    return tf.keras.applications.VGG19(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )

def get_resnet50(): 
    return tf.keras.applications.resnet.ResNet50(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )

def get_inceptionv3(): 
    return tf.keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3),weights='imagenet',include_top=False )

def get_mobilenetV2(): 
    return tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )

def get_mobilenetV1(): 
    return tf.keras.applications.mobilenet.MobileNet(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    
def get_xception(): 
    return tf.keras.applications.xception.Xception(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )

def get_efficientnetb0(): 
    return tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )



def create_model(pretrained_model): 
    
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

    return pretrained_model.name, model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def display_plot_per_history(histories,folder):
    # summarize history for accuracy
    for name, history in histories:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'accuracy for {name} over epochs')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.savefig(os.path.join('results',folder,'graphs',f'accuracy{name}.png'))
        
        plt.clf() 
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'loss for {name} over epochs')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(os.path.join('results',folder,'graphs',f'loss{name}.png'))
        plt.clf()

def display_plot_for_histories(histories,folder):
    # summarize history for accuracy
    for name,history in histories:
        plt.plot(history.history['accuracy'])
    plt.title('models training accuracy over epochs')
    plt.ylabel('training accuracy')
    plt.xlabel('epoch')
    plt.legend([name for name,history in histories], loc='lower right')
    plt.savefig(os.path.join('results',folder,'general_graphs','accuracy.png'))
    plt.clf() 
    # summarize history for loss
    for name,history in histories:
        plt.plot(history.history['loss'])
    plt.title('models training loss over epochs ')
    plt.ylabel('training loss')
    plt.xlabel('epoch')
    plt.legend([name for name,history in histories],loc='upper right')
    plt.savefig(os.path.join('results',folder,'general_graphs','loss.png'))
    plt.clf() 

def run_cross_validation_create_models(pretrained_model,X_train,Y_train,X_valid,Y_valid,X_test,Y_test,folder):
    # input image dimensions
    
    num_epoch = 80
    
    
    #X_train, Y_train, train_id = read_and_normalize_train_data()
    #X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)

    name, model = create_model(pretrained_model)
    
    train_generator = DataGenerator(X_train,Y_train, 8)
    validate_generator = DataGenerator(X_valid,Y_valid, 8)
    logs_path = os.path.join('results',folder,'info')
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logs_path,f'logs_{name}.csv'), append=False, separator=';')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3,min_delta=0.01)
    history = model.fit(
        train_generator,
        epochs=num_epoch,
        validation_data=validate_generator,
        callbacks=[callback,csv_logger]
    )
    #test data 
    #Y_test = Y_valid
    #X_test, Y_test = read_and_normalize_test_data()
    validate_generator = DataGenerator(X_test, Y_test, 8)
    
    
    results = model.evaluate(validate_generator)
    predictions = model.predict(validate_generator)

    Y_test = np.argmax(Y_test,axis=1)
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(Y_test,predictions_labels)
    print(conf_matrix)
    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)
    with open(os.path.join(logs_path,f'test_info_{name}.txt'), 'w') as f:
        f.write(np.array2string(confusion_matrix(Y_test,predictions_labels), separator=', '))
        
        f.write(f'\naciertos: {np.trace(conf_matrix)}\n')
        f.write(f'total: {np.sum(conf_matrix)}\n')
        f.write(f'test loss, test acc: {results}\n' )
        
    return name,history

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    models = [get_vgg19,get_resnet50,get_inceptionv3,get_mobilenetV2,get_mobilenetV1,get_xception,get_efficientnetb0]
    #models=[get_vgg19]
    histories = []
    
    folder = 'NatureConservancy'    
    #folder = 'NatureConservancyCropped'    
    #folder = 'FishSpecies'
    
    X_train, Y_train, train_id = read_and_normalize_train_data(folder)    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)

    #to test identification
    X_test, Y_test = X_valid,Y_valid    
    
    #X_test, Y_test = read_and_normalize_test_data(folder)

    for model in models:
        name, history = run_cross_validation_create_models(model(),X_train,Y_train,X_valid,Y_valid,X_test,Y_test,folder)
        histories.append((name,history))
        tf.keras.backend.clear_session()

    display_plot_per_history(histories,folder)
    display_plot_for_histories(histories,folder)
    

    #run_cross_validation_process_test(info_string, models)
