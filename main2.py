

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
seed = 2022
np.random.seed(seed)
import os
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, log_loss,balanced_accuracy_score,recall_score,f1_score,accuracy_score
#from keras import __version__ as keras_version
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

from keras import backend as K
from utils.utils import f1_m,precision_m, recall_m

print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
columns=['ALB','BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'OTHER']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone','Trout']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=columns)
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data(folder):
    #path = os.path.join('.',folder)
    path = os.path.join('.',folder, 'train')

    batch_size = 32

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1
    )
    train_ds = datagen.flow_from_directory(
            directory = path,
            target_size=(IMG_COUNT,IMG_COUNT),
            batch_size=batch_size,
            class_mode='categorical',
            #seed = seed,
            subset='training',
            shuffle=True)
    val_ds = datagen.flow_from_directory(
            directory = path,
            target_size=(IMG_COUNT,IMG_COUNT),
            batch_size=batch_size,
            class_mode='categorical',
            #seed = seed,
            subset='validation',
            shuffle=False)

    return train_ds,val_ds

def read_and_normalize_test_data(folder):
    path = os.path.join('.',folder, 'test')

    batch_size = 32

    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_ds = datagen.flow_from_directory(
            directory = path,
            target_size=(IMG_COUNT,IMG_COUNT),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    

    return test_ds



def create_model(): 
    #tf.keras.applications.efficientnet.EfficientNetB0(include_top=True).summary()
    
    #pretrained_model = tf.keras.applications.VGG19(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.resnet.ResNet50(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.xception.Xception(input_shape=(299,299,3),weights='imagenet',include_top=False )
    #for efficientnets pls check: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.efficientnet.EfficientNetB2(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    pretrained_model.summary()
    
    #pretrained_model.trainable = False
    for i, layer in enumerate(pretrained_model.layers):
        if (layer.name.startswith('block7')) and not isinstance(layer, tf.keras.layers.BatchNormalization):
            print(i, layer.name)
            layer.trainable = True
        else:
            layer.trainable = False 

    print(pretrained_model.output)
    layers = [1024,256,64,len(columns)]
    sequential_layers = [pretrained_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Flatten()]
    top_dropout_rate = 0.5 
    for layer in layers:
        sequential_layers.append(tf.keras.layers.BatchNormalization())
        sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
        if layer != layers[-1]:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="relu"))    
        else:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="softmax", name="pred"))
    model = tf.keras.Sequential(sequential_layers)
    model.summary()
    
    optimizer = tf.keras.optimizers.RMSprop(momentum=0.0001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=['accuracy',f1_m,precision_m, recall_m],

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return pretrained_model.name,model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def display_plot_for_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')


def run_cross_validation_create_models():
    print(tf.__version__)
    # input image dimensions
    num_epoch = 100
    folder = 'NatureConservancyCropped'
    #folder = 'NatureConservancy'
    train_ds,val_ds = read_and_normalize_train_data(folder)
    #test_ds = read_and_normalize_test_data(folder)
    test_ds = val_ds

    #train_ds,val_ds = read_and_normalize_train_data('FishSpecies')
    #test_ds = read_and_normalize_test_data('FishSpecies')
    
    name,model = create_model()
    

    print(test_ds.class_indices)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(
        train_ds,
        epochs=num_epoch,
        validation_data=val_ds,
        callbacks=[callback]
        #callbacks=[]
    )
    display_plot_for_history(history)
    #test data 
    model.save('./models/model.h5')
    predictions = model.predict(test_ds)
    

    results = model.evaluate(test_ds)

    #extract labels
    Y_test = test_ds.classes
    
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(predictions_labels,Y_test)

    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)
    print("accuracy score is: ",accuracy_score(Y_test,predictions_labels  ))
    print("balanced accuracy score is: ",balanced_accuracy_score(Y_test,predictions_labels))
    print("f1-weighted score is: ",f1_score(Y_test,predictions_labels,average='weighted'))
    conf_matrix = conf_matrix *100/ conf_matrix.astype(float).sum(axis=0)
    conf_matrix = np.round(conf_matrix,decimals=3)    
    print(conf_matrix)
    np.savetxt("experimento1_conf_matrix.csv",conf_matrix,delimiter=",")

if __name__ == '__main__':
    run_cross_validation_create_models()
