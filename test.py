<<<<<<< HEAD
import cv2
import argparse
=======
'''import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
>>>>>>> 4b471a066cc76698a4c94a4446a108b736ae14f7
import numpy as np
from operator import itemgetter 

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

import numpy as np
import cv2
import time
from utils.fps import FPS
from utils.webcam import WebcamVideoStream
from threading import Thread


print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("GPU_CUDA: ", cv2.cuda.getCudaEnabledDeviceCount())
print(cv2.__version__)
device_name = tf.test.gpu_device_name()
print(device_name)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def draw_prediction(img, color, label , confidence, x, y, x_plus_w, y_plus_h):

    label = label + str(confidence)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



scale = 1.0/255

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

columns=np.array(['ALB','BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
#COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
COLORS = np.random.uniform(0, 255, size=(len(columns), 3))


model = tf.keras.models.load_model('model.h5')

net = cv2.dnn.readNet(args.weights, args.config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


conf_threshold = 0
nms_threshold = 0.6
#nms_threshold = 0.9
#conf_threshold = 0.5
#nms_threshold = 0.45
IMG_SIZE = (224,224)

cap = WebcamVideoStream(src=0).start()
fps = FPS()

class_ids = []
confidences = []
boxes = []
while True:
    fps.start()
    for i in range(120):
        image = cap.read()
        cv2.imshow('img', image)
        if cv2.waitKey(30) & 0xff == ord('q'):
            exit()
        fps.update()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    



<<<<<<< HEAD
=======
#print(classification_report(T4, T5,target_names=['0','1']))
'''
import os
from pathlib import Path
import imghdr
>>>>>>> 4b471a066cc76698a4c94a4446a108b736ae14f7

def check_images_types(dir):
	image_extensions = [".png", ".jpg"]  # add there all your images file extensions

	img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
	for filepath in Path(dir).rglob("*"):
		if filepath.suffix.lower() in image_extensions:
			img_type = imghdr.what(filepath)
			if img_type is None:
				print(f"{filepath} is not an image")
			elif img_type not in img_type_accepted_by_tf:
				print(f"{filepath} is a {img_type}, not accepted by TensorFlow")


train_dir = os.path.join('FishSpecies','train')
test_dir = os.path.join('FishSpecies','test')

check_images_types(train_dir)
check_images_types(test_dir)
