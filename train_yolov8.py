from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
 
model.train(data="Objects365.yaml")  # train the model
model.val()  # evaluate model performance on the validation set
model.predict(source="https://ultralytics.com/images/bus.jpg")  # predict on an image
model.export(format="onnx")

#have to run in other disk
