

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
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
seed = 2022
np.random.seed(seed)
import os
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,recall_score,f1_score,accuracy_score
from keras import __version__ as keras_version
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
from keras_dataloader.datagenerator import DataGenerator
from utils.utils import generate_data_csv


print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
columns=['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone','Trout']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    

def create_model(): 
    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model.summary()
    
    pretrained_model.trainable = False

    layers = [1024,256,64,len(columns)]
    sequential_layers = [pretrained_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Flatten()]
    #sequential_layers = [pretrained_model,tf.keras.layers.Flatten()]
    top_dropout_rate = 0.5 
    for layer in layers:
        sequential_layers.append(tf.keras.layers.BatchNormalization())
        sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
        if layer != layers[-1]:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="relu"))    
        else:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="softmax", name="pred"))
    model = tf.keras.Sequential(sequential_layers)
    #model.summary()
    
    optimizer = tf.keras.optimizers.RMSprop()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=['accuracy'],

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return model


def display_plot_for_history(history,k):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join('Kfolds',str(k),'accuracy.png'))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join('Kfolds',str(k),'loss.png'))
    plt.clf()

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

def predict(model,ds):
    predictions = model.predict(ds)

    results = model.evaluate(ds)

    #extract labels
    Y_test = ds.classes
    
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(predictions_labels,Y_test)
    print(conf_matrix)
    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)
    conf_matrix = conf_matrix / conf_matrix.astype(float).sum(axis=0)
    return predictions_labels,conf_matrix,results

def predict_and_save_results(model,train_ds,val_ds,k,results_writer):
    _,train_conf_matrix,results_train = predict(model,train_ds)
    predictions_labels,val_conf_matrix,results_val = predict(model,val_ds)
    Y_test = val_ds.classes
    with open(os.path.join('Kfolds',str(k),'test_info.txt'), 'w') as f:
        f.write(np.array2string(train_conf_matrix, separator=', '))
        f.write(f'train loss, train acc: {results_train}\n' )
        f.write(np.array2string(val_conf_matrix, separator=', '))
        f.write(f'train loss, train acc: {results_val}\n' )
        f.write(f'accuracy score is: {accuracy_score(Y_test,predictions_labels)}')
        f.write(f'balanced accuracy score is: {balanced_accuracy_score(Y_test,predictions_labels)}')
        f.write(f'f1-weighted score is: {f1_score(Y_test,predictions_labels,average="weighted")}')
    results_writer.writerow([str(k),str(results_train[0]),results_train[1],results_val[0],results_val[1]])



#######################
generate_data_csv('NatureConservancyCropped',['train'])

train_data = pd.read_csv('data.csv')
Y = train_data[['label']]

                         
skf = StratifiedKFold(n_splits = 10, shuffle = False) 

idg = tf.keras.preprocessing.image.ImageDataGenerator()


VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

save_dir = '/saved_models/'
fold_var = 1
n = len(train_data)
image_dir = '.'
num_epochs = 100
batch_size = 32
with open('results.csv','w',newline='') as results_file:
    writer = csv.writer(results_file, delimiter=',')
    writer.writerow(["iter","train_loss","train_acc","val_loss","val_acc"])
    for train_index, val_index in skf.split(np.zeros(n),Y):
        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]
        print(len(training_data))
        print(len(validation_data))
            
        train_data_generator = idg.flow_from_dataframe(training_data, directory = image_dir,
                                x_col = "filename", y_col = "label",
                                class_mode = "categorical", shuffle = False,
                                target_size=(IMG_COUNT,IMG_COUNT),batch_size=batch_size)
            
        valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir,
                                x_col = "filename", y_col = "label",
                                class_mode = "categorical", shuffle = False,
                                target_size=(IMG_COUNT,IMG_COUNT),batch_size=batch_size)
        
        # CREATE NEW MODEL
        model = create_model()
        # CREATE CALLBACKS
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        # FIT THE MODEL
        
        history = model.fit(train_data_generator,
                epochs=num_epochs,
                callbacks=[callback],
                validation_data=valid_data_generator)
        display_plot_for_history(history,fold_var)
        # LOAD BEST MODEL to evaluate the performance of the model
        model.save("/Kfolds/model_"+str(fold_var)+".h5")
        
        results = model.evaluate(valid_data_generator)

        predict_and_save_results(model,train_data_generator,valid_data_generator,fold_var,writer)
        
        tf.keras.backend.clear_session()
        fold_var += 1
