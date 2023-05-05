import cv2
import argparse
import numpy as np
import glob
import shutil
import os
import random
import math
from utils import copy_folder_to_array

random.seed(42069)

columns=['ALB','BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'OTHER']

image_files = []
for i in columns:
    folder = f'./NatureConservancy/train/{i}/*.jpg'
    folder_size = copy_folder_to_array(folder,image_files)
    print(i,str(folder_size))

print("total:",len(image_files))
copy_folder_to_array(f'./NatureConservancy/test_stg1/*.jpg',image_files)
print("total:",len(image_files))
random.shuffle(image_files)
image_files = image_files[0:int(len(image_files)/5)]
print("total:",len(image_files))

for folder in columns: 
    try:
        shutil.rmtree(f'./testingPipeline/yolo/{folder}')
        os.mkdir(f'./testingPipeline/yolo/{folder}')
    except:
        os.mkdir(f'./testingPipeline/yolo/{folder}')

for image_file in image_files:
    image_file = image_file.replace('/','\\')
    dataset = image_file.split('\\')[2]
    image = cv2.imread(image_file)
    #print(image_file)
    if dataset =="train":
        col = image_file.split('\\')[3]
        name = image_file.split('\\')[4]
        cv2.imwrite(f'./testingPipeline/yolo/{col}/{name}', image)
    else:
        name = image_file.split('\\')[3]
        cv2.imwrite(f'./testingPipeline/yolo/missing/{name}', image)

