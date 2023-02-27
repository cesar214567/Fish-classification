from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov5m_Objects365.pt")  # load a pretrained model (recommended for training)

# Use the model
success = model.export(format="onnx",optimize = optimize_gpu)  # export the model to ONNX format