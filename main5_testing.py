import cv2
import argparse
import numpy as np
from operator import itemgetter 
import glob
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
import tensorflow as tf
import os
from utils.utils import get_folders_size
from utils.utils import generate_data_csv,f1_m,precision_m, recall_m,get_folders_size,get_folders_cropped_images_size
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,recall_score,f1_score,accuracy_score,precision_score


ap = argparse.ArgumentParser()

ap.add_argument('-dataset','--dataset', required=True,
                help = 'name of input dataset')

args = ap.parse_args()

IMG_COUNT = 224
columns=['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
columns_without_crop=['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT','errors','missing']
folder = f'./testingPipeline/'
cropped_images_folder = f'{folder}{args.dataset}/'
images_folder = f'{folder}images/'

print(get_folders_size(cropped_images_folder,columns=columns_without_crop))
print(get_folders_cropped_images_size(images_folder,columns=columns_without_crop))

model = tf.keras.models.load_model('./models/model.h5',custom_objects={"f1_m":f1_m,"precision_m":precision_m, "recall_m":recall_m})

batch_size = 32
datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = datagen.flow_from_directory(
            directory = cropped_images_folder,
            classes= columns,
            target_size=(IMG_COUNT,IMG_COUNT),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    

predictions = model.predict(test_ds)
    

results = model.evaluate(test_ds)

    #extract labels
Y_test = test_ds.classes
    
with open(os.path.join(f'{args.dataset}_info.txt'), 'w')as f :
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(Y_test,predictions_labels)
    f.write(f'accuracy score is: {accuracy_score(Y_test,predictions_labels)}\n')
    f.write(f'balanced accuracy score is: {balanced_accuracy_score(Y_test,predictions_labels)}\n')
    f.write(f'f1-weighted score is: {f1_score(Y_test,predictions_labels,average="weighted")}\n')
    f.write(f'precision score is: {precision_score(Y_test,predictions_labels,average="weighted")}\n')
    f.write(f'recall score is: {recall_score(Y_test,predictions_labels,average="weighted")}')

    print("accuracy score is: ",accuracy_score(Y_test,predictions_labels))
    print("balanced accuracy score is: ",balanced_accuracy_score(Y_test,predictions_labels))
    print("weighted f1 score is: ",f1_score(Y_test,predictions_labels,average='weighted'))
    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)

    conf_matrix = conf_matrix *100/ conf_matrix.astype(float).sum(axis=0)
    conf_matrix = np.round(conf_matrix,decimals=3)    
    print(conf_matrix)
    np.savetxt(f'{args.dataset}_conf_matrix.csv',conf_matrix,delimiter=",")

