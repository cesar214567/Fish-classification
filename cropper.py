

import cv2
import argparse
import numpy as np
import glob
import shutil
import os
columns=['ALB','BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'OTHER']

def crop_images(folder_from,folder_to,labeled=False):
    for folder in columns: 
        shutil.rmtree(f'./{folder_to}/{folder}')
        os.mkdir(f'./{folder_to}/{folder}')

    for img_path in glob.glob(f'./{folder_from}/*.jpg'):
        image = cv2.imread(img_path)
        img_labels = img_path.replace(".jpg",".txt")
        Width = image.shape[1]
        Height = image.shape[0]
        print(Width,Height)
        scale = 1.0/255
        print(img_labels)
        with open(img_labels,'r') as f:
            counter = 0
            for line in f.readlines():
                counter+=1
                items = line.split()
                class_id = int(items[0])
                center_x = int(float(items[1]) * Width)
                center_y = int(float(items[2]) * Height)
                w = int(float(items[3]) * Width)
                h = int(float(items[4]) * Height)
                x = round(center_x - w / 2)
                y = round(center_y - h / 2)
                print(x,y,x+w,y+h)
                #draw_prediction(image,class_id,1,x,y,x+w,y+h)
                fish_image = image[y:y+h,x:x+w]
                #image = cv2.resize(image, (1200,800), interpolation = cv2.INTER_AREA)
                img_path_params = img_path.split("\\")
                if labeled:
                    type = img_path_params[1]
                    image_number = img_path_params[2].removesuffix('.jpg')
                    try:
                        cv2.imwrite(f'./{folder_to}/{type}/{image_number}_{counter}.jpg', fish_image)
                    except:
                        cv2.imshow("object detection", fish_image)
                else:
                    type = columns[class_id]
                    image_number = img_path_params[1].removesuffix('.jpg')
                    try:
                        cv2.imwrite(f'./{folder_to}/{type}/{image_number}_{counter}.jpg', fish_image)
                    except:
                        cv2.imshow("object detection", fish_image)
                print(type)
                

crop_images('NatureConservancy/train/*','NatureConservancyCropped/train',labeled=True)
crop_images('NatureConservancy/test_stg1','NatureConservancyCropped/test')
