import os
import glob
import random
import csv
from keras import backend as K

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


def generate_data_csv(folder, datasets =["train"]):
    columns=['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    data = []
    for label in columns:
        for dataset in datasets:
            path = os.path.join('.',folder,dataset,label,'*.jpg')
            files = glob.glob(path)
            for fl in files:
                #print(fl)
                flbase = os.path.basename(fl)
                data.append([fl,label])
    random.shuffle(data)
    with open(os.path.join("data.csv"),'w',newline='') as f :
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['filename','label'])
        for line in data:
            writer.writerow(line)

def copy_folder_to_array(folder,images):
    globs = glob.glob(folder)
    for image_file in globs:
        images.append(image_file)
    return len(globs)
