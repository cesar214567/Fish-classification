

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
import math
seed = 2022
np.random.seed(seed)
import os
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
from PIL import Image

import numpy as np
from keras_dataloader.datagenerator import DataGenerator

from keras import backend as K
import shutil

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
columns=['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone','Trout']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)
#print("device is: ",device)
#torch.cuda.set_per_process_memory_fraction(1.0, device=None) 

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
            shuffle=False)
    val_ds = datagen.flow_from_directory(
            directory = path,
            target_size=(IMG_COUNT,IMG_COUNT),
            batch_size=batch_size,
            class_mode='categorical',
            #seed = seed,
            subset='validation',
            shuffle=False)

    return train_ds,val_ds

def get_max_images_count(folder,subfolder="train"):
    images_count=[]
    count = 0
    for label in columns:
        images = len(os.listdir(os.path.join(folder,subfolder,label)))
        images_count.append(images)
        count += images
    max_count = images_count[np.argmax(images_count)]
    return max_count,images_count, count

def augment():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomBrightness((-0.1,0.1)),
        tf.keras.layers.RandomContrast(0.05),
        tf.keras.layers.RandomZoom((0.0,0.25)),
        tf.keras.layers.Rescaling(0.2)

    ])
    folder = 'NatureConservancyCropped'
    folder_to = 'NatureConservancyCroppedAugmented'
    for column in columns: 
        shutil.rmtree(f'./{folder_to}/train/{column}')
        os.mkdir(f'./{folder_to}/train/{column}')
    train_ds,val_ds = read_and_normalize_train_data(folder)
    max_count,images_count,count = get_max_images_count(folder)
    batch_number = 0
    for x_batch,y_batch in train_ds:
        if batch_number > count/32:
            exit()
        batch_number+=1
        predictions_labels = np.argmax(y_batch,axis=1)
        for image_id,(image,class_id) in enumerate(zip(x_batch,predictions_labels)):
            times = math.ceil(float(max_count)/images_count[class_id])
            print(times)
            if times > 4:
                times = 4
            for i in range(times):
                augmented_image = data_augmentation(image)
                augmented_image = np.array(augmented_image, dtype=np.uint8)
                print(columns[class_id])
                img = Image.fromarray(np.uint8(augmented_image)).convert('RGB')
                #plt.show()
                img.save(os.path.join('.',folder_to,'train',columns[class_id],str(batch_number)+"_"+str(image_id)+"_"+str(i)+".jpg"))

if __name__ == "__main__":
    folder_to = 'NatureConservancyCropped'
    subfolder = "test"
    max_count,images_count,amount = get_max_images_count(folder_to,subfolder)
    print(max_count,images_count,amount)
    #augment()

    






