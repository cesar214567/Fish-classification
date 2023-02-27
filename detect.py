import cv2
import argparse
import numpy as np

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

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id]) + str(confidence)
    print(label)
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
image = cv2.imread(args.image)

Width = image.shape[1]
Height = image.shape[0]
scale = 1.0/255

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

print(COLORS)

net = cv2.dnn.readNet(args.weights, args.config)

SIZE = 1280
#SIZE = 224
#blob = cv2.dnn.blobFromImage(image, scale, (1280,704), [0,0,0],1, crop=False)
blob = cv2.dnn.blobFromImage(image, scale, (800,608), [0,0,0],1, crop=False)
#blob = cv2.dnn.blobFromImage(image, scale, (384,288), [0,0,0],1, crop=False)

net.setInput(blob)

outs = net.forward(net.getUnconnectedOutLayersNames())
print(np.array(outs[0]).shape)
print(np.array(outs[0]))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0
nms_threshold = 0.6
#nms_threshold = 0.9
#conf_threshold = 0.5
#nms_threshold = 0.45

print(outs[0])
print('\n')
print(outs[1])
print('\n')
print(outs[2])
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.00:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

print(len(boxes))
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print(len(indices))
for i in indices:
    try:
        box = boxes[i]
    except:
        i = i[0]
        box = boxes[i]
    
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], round(confidences[i],2), round(x), round(y), round(x+w), round(y+h))
image = cv2.resize(image, (1200,800), interpolation = cv2.INTER_AREA)

cv2.imshow("object detection", image)
cv2.waitKey()
    
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
