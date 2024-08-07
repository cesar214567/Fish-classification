import cv2
import argparse
import numpy as np
from operator import itemgetter 
from utils.utils import generate_data_csv,f1_m,precision_m, recall_m

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

IMG_SIZE = (224,224)
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=False,
                help = 'path to yolo config file')
ap.add_argument('-wD', '--weightsDetector', required=True,
                help = 'path to detector pre-trained weights')
ap.add_argument('-clD', '--classesDetector', required=True,
                help = 'path to text file containing class names for detector')
ap.add_argument('-wC', '--weightsClassifier', required=True,
                help = 'path to CNN pre-trained weights')
ap.add_argument('-clC', '--classesClassifier', required=True,
                help = 'path to text file containing class names for CNN')
ap.add_argument('-expand', '--expand', required=True,
                help = 'expand image?(for pretrained yolov5)')
args = ap.parse_args()

def draw_prediction(img, color, label , confidence, x, y, x_plus_w, y_plus_h):
    label = label + str(confidence)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)


scale = 1.0/255

classes = None
columns = None
with open(args.classesDetector, 'r') as f:
    classes = np.array([line.strip() for line in f.readlines()])

with open(args.classesClassifier, 'r') as f:
    columns = np.array([line.strip() for line in f.readlines()])

#columns=np.array(['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT'])
#columns=np.array(['ALB','BET', 'DOL', 'LAG','MugilCephalus', 'OTHER','RhinobatosCemiculus','ScomberJaponicus','SHARK','TetrapturusBelone','Trout', 'YFT'])
#COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
COLORS = np.random.uniform(0, 255, size=(len(columns), 3))


model = tf.keras.models.load_model(args.weightsClassifier,custom_objects={"f1_m":f1_m,"precision_m":precision_m, "recall_m":recall_m})
type = ""
if args.weightsDetector.endswith('.onnx'):
    net = cv2.dnn.readNet(args.weightsDetector)
    type = "ONNX"
elif args.weightsDetector.endswith('.weights'):
    net = cv2.dnn.readNet(args.weightsDetector, args.configDetector)
    type = "WEIGHTS"
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
SIZE = 1280
#SIZE = 224
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
conf_threshold = 0.04
nms_threshold = 0.05
#for yolov5 not trained 
fish_conf_threshold = 0.4
#nms_threshold = 0.9
#conf_threshold = 0.5
#nms_threshold = 0.45
def relu(x):
    return max(0.0, x)


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
    if args.expand =='yes':
        x = max(0,round(x - w*0.2))
        y = max(0,round(y - h*0.2))
        w = min(image.shape[1],round(w+w*0.4))
        h = min(image.shape[0],round(h+h*0.4))
    class_id = class_ids[i]
    label = str(classes[class_id])


    if (("best" in args.weightsDetector and confidences[i] >= fish_conf_threshold) or (label!="Fishing Rod" and ("Fish" in label or "Seafood" in label )and confidences[i] >= fish_conf_threshold)):
        print(label)
        fish_image = image[y:y+h,x:x+w]
        fish_image = cv2.resize(fish_image, IMG_SIZE, cv2.INTER_LINEAR)
        fish_image = fish_image.reshape(224,224,3)
        fish_images.append(fish_image)
        fish_info.append({"index":i,"x":x,"y":y,"w":w,"h":h})
        #print(label,confidences[i])
        #draw_prediction(image,[0,0,0], label, confidences[i], x, y, x+w, y+h)
    else:
        #print(label,confidences[i])
        confidence = round(confidences[i],2)
        draw_prediction(image,[0,0,0], label, confidence, x, y, x+w, y+h)


if len(fish_images)!=0:
    fish_images = np.array(fish_images).reshape(len(fish_images),224,224,3)
    prediction = model.predict(fish_images)
    fish_ids = np.argmax(prediction,axis=1)
    labels = columns[fish_ids]
    colors = COLORS[fish_ids]
    confidences = np.round(np.max(prediction,axis=1),decimals=2)
    for i,info in enumerate(fish_info):
        index,x,y,w,h = info['index'],info['x'],info['y'],info['w'],info['h']
        label = labels[i]
        color = colors[i]
        confidence = confidences[i]
        if confidence >= fish_conf_threshold:
            draw_prediction(image,color, label, confidence, x, y, x+w, y+h)
            print(label,confidence)

image = cv2.resize(image, (1200,800), interpolation = cv2.INTER_LINEAR)

cv2.imshow("object detection", image)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
