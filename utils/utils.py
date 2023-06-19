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

def copy_images_to_array(folder, columns=[]):
    array = []
    sum = 0
    for i in columns:
        folder = f'{folder}{i}/*.jpg'
        folder_size = copy_folder_to_array(folder,array)
        sum+= folder_size
        print(i,str(folder_size))
    print(sum)


def get_folders_size(folder,columns=[]):
    sum = 0
    folders={}
    for i in columns:
        glob_path = f'{folder}{i}/*.jpg'
        folder_size = len(glob.glob(glob_path))
        sum += folder_size
        folders[i] = folder_size
    return sum,folders
    
def get_folders_cropped_images_size(folder, columns=[]):
    sum = 0
    folders=dict(zip(columns,[0]*len(columns)))
    for i in columns:
        folder_image_size = 0
        glob_path = f'{folder}{i}/*.txt'
        labels_paths = glob.glob(glob_path)
        for label_path in labels_paths:
            with open(label_path,'r') as f:
                lines = f.readlines()
                folder_image_size += len(lines)
                if i == 'missing':
                    for line in lines:
                        folders[columns[int(line.split(' ')[0])]]+=1
        sum += folder_image_size
        folders[i] += folder_image_size
    return sum,folders