import cv2
import argparse
import numpy as np
from operator import itemgetter 
from utils.utils import generate_data_csv,f1_m,precision_m, recall_m
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
import shutil
import os 
import math
IMG_SIZE = (224,224)

IMG_SIZE = (224,224)
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


ap = argparse.ArgumentParser()

ap.add_argument('-c', '--config', required=False,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-out', '--output', required=True,
                help = 'name of output folder')

args = ap.parse_args()
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

columns=np.array(['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT','missing'])
#COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
COLORS = np.random.uniform(0, 255, size=(len(columns), 3))
def relu(x):
    return max(0.0, x)




scale = 1.0/255

type = ""
if args.weights.endswith('.onnx'):
    net = cv2.dnn.readNet(args.weights)
    type = "ONNX"
elif args.weights.endswith('.weights'):
    net = cv2.dnn.readNet(args.weights, args.config)
    type = "WEIGHTS"
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
SIZE = 1280
#SIZE = 224
conf_threshold = 0.02
nms_threshold = 0.05
#for yolov5 not trained 
fish_conf_threshold = 0.15

#for yolov5 trained
#fish_conf_threshold = 0.95
#fish_conf_threshold = 0.95


def map_boxes(detection):
    scores = detection[5:]
    confidence =detection[4]
    class_id = np.argmax(scores)
    if confidence > conf_threshold:
        confidence = scores[class_id]
        
        center_x = int(detection[0] * Width)
        center_y = int(detection[1] * Height)
        w = int(relu(detection[2] * Width))
        h = int(relu(detection[3] * Height))
        x = int(relu(center_x - w / 2))
        y = int(relu(center_y - h / 2))
        class_ids.append(class_id)
        confidences.append(float(confidence))
        boxes.append([x, y, w, h])

    


for folder in columns: 
    try:
        shutil.rmtree(f'./testingPipeline/{args.output}/{folder}')
        os.mkdir(f'./testingPipeline/{args.output}/{folder}')
    except:
        os.mkdir(f'./testingPipeline/{args.output}/{folder}')

counter = 0
for col in columns:
    for image_file in glob.glob(f'./testingPipeline/images/{col}/*.jpg'):
        image = cv2.imread(image_file)
        print(image_file)
        image_name = image_file.replace('/','\\').split('\\')[4]

        Width = image.shape[1]
        Height = image.shape[0] 
        if type=="WEIGHTS":
            #blob = cv2.dnn.blobFromImage(image, scale, (1280,704), [0,0,0],1, crop=False)
            blob = cv2.dnn.blobFromImage(image, scale, (800,608), [0,0,0],1, crop=False)
            #blob = cv2.dnn.blobFromImage(image, scale, (384,288), [0,0,0],1, crop=False)
        elif type=="ONNX":    
            Width = Width/640.
            Height = Height/640.
            blob = cv2.dnn.blobFromImage(image, scale, (640,640), [0,0,0],1, crop=False)

                
        net.setInput(blob)

        outs = net.forward(net.getUnconnectedOutLayersNames())

        class_ids = []
        confidences = []
        boxes = []


        if type=="ONNX":
            outs=outs[0]

        for detection in outs[0]:
            map_boxes(detection)
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        fish_info = []
        fish_images = []
        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]

            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])
            x = max(0,round(x - w*0.15))
            y = max(0,round(y - h*0.15))
            w = round(w+w*0.3)
            h = round(h+h*0.3)
            class_id = class_ids[i]
            label = str(classes[class_id])
            if (args.weights.startswith("best") and confidences[i] >= fish_conf_threshold) or (label!="Fishing Rod" and ("Fish" in label or "Seafood" in label )and confidences[i] >= fish_conf_threshold):
                print(counter,str(confidences[i]))
                height,width, channels = image.shape
                fish_image = image[y:min(height,y+h),x:min(width,x+w)]
                #fish_image = cv2.resize(fish_image, IMG_SIZE, cv2.INTER_LINEAR)
                cv2.imwrite(f'./testingPipeline/{args.output}/{col}/{str(counter)}.jpg', fish_image)
                counter+=1